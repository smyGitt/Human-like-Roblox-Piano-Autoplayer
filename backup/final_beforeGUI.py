#!/usr/bin/env python3
#
# MIDI-to-Keyboard Player: Final Version
# A robust, clear, and error-free implementation with advanced humanization
# and statistically-driven automatic pedal generation.
#
import argparse
import mido
import time
import heapq
import signal
import threading
import random
import copy
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
from pynput.keyboard import Key, Controller

# New dependency for advanced section analysis
try:
    from sklearn.cluster import KMeans
except ImportError:
    print("Scikit-learn not found. Please run 'pip install scikit-learn' for advanced pedal analysis.")
    exit(1)


# --- Core Data Structures ---

@dataclass
class Note:
    """Represents a single, parsed musical note."""
    pitch: int
    velocity: int
    start_time: float
    duration: float
    hand: str = 'unknown'

@dataclass(order=True)
class KeyEvent:
    """Represents a scheduled keyboard event for the priority queue."""
    time: float
    priority: int = field(compare=True)
    action: str = field(compare=False)
    key_char: str = field(compare=False)

@dataclass
class RhythmicPhrase:
    """Represents a sub-section with a consistent rhythmic and melodic character."""
    start_time: float
    end_time: float
    notes: List[Note]
    articulation_label: str
    pattern_label: str = 'standard'

@dataclass
class MusicalSection:
    """Represents a major musical phrase, which contains smaller rhythmic phrases."""
    start_time: float
    end_time: float
    note_count: int
    notes: List[Note]
    normalized_density: float
    pace_label: str = 'unclassified'
    rhythmic_phrases: List[RhythmicPhrase] = field(default_factory=list)
    is_bridge: bool = False

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

@dataclass
class KeyState:
    """Tracks the complete state of a single keyboard key."""
    key_char: str
    is_active: bool = False   # Is a note event currently holding the key down? ("finger press")
    is_sustained: bool = False # Has the note been released but the pedal is holding it?

    def press(self):
        self.is_active = True

    def release(self, pedal_is_down: bool):
        self.is_active = False
        if pedal_is_down:
            self.is_sustained = True
        else:
            self.is_sustained = False

    def lift_sustain(self):
        self.is_sustained = False

    @property
    def is_physically_down(self) -> bool:
        return self.is_active or self.is_sustained
        
@dataclass
class Finger:
    """Represents the state of a single physical finger."""
    id: int  # 0-4: Left Thumb to Pinky; 5-9: Right Thumb to Pinky
    hand: str
    current_pitch: Optional[int] = None
    last_press_time: float = -1.0

class TempoMap:
    """Stores all tempo changes and provides a lookup for any point in time."""
    def __init__(self, tempo_events: List[Tuple[float, int]]):
        self.events = sorted(tempo_events, key=lambda x: x[0])

    def get_tempo_at(self, time: float) -> int:
        if not self.events: return 500000
        for event_time, tempo in reversed(self.events):
            if time >= event_time: return tempo
        return self.events[0][1]

# --- Component Classes ---

class MidiParser:
    """Parses a MIDI file into a simple list of Note objects and a TempoMap."""
    @staticmethod
    def parse(filepath: str, tempo_scale: float = 1.0, debug: bool = False) -> Tuple[List[Note], List[Tuple[float, int]]]:
        try: mid = mido.MidiFile(filepath)
        except Exception as e: raise IOError(f"Could not read or parse MIDI file: {e}")
        notes: List[Note] = []
        tempo_map_data: List[Tuple[float, int]] = []
        open_notes: Dict[int, List[Dict]] = defaultdict(list)
        absolute_time: float = 0.0
        tempo = 500000
        ticks_per_beat = mid.ticks_per_beat or 480
        tempo_map_data.append((0.0, tempo))
        for msg in mido.merge_tracks(mid.tracks):
            absolute_time += mido.tick2second(msg.time, ticks_per_beat, tempo)
            if msg.type == 'set_tempo':
                tempo = msg.tempo
                tempo_map_data.append((absolute_time, tempo))
            elif msg.type == 'note_on' and msg.velocity > 0:
                open_notes[msg.note].append({'start': absolute_time, 'vel': msg.velocity})
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                if open_notes[msg.note]:
                    note_data = open_notes[msg.note].pop(0)
                    start = note_data['start']
                    duration = absolute_time - start
                    if duration > 0.01:
                        notes.append(Note(pitch=msg.note, velocity=note_data['vel'], start_time=start / tempo_scale, duration=duration / tempo_scale))
        notes.sort(key=lambda n: n.start_time)
        return notes, tempo_map_data

class KeyMapper:
    """Maps MIDI pitches to keyboard keys."""
    LAYOUT = "1!2@34$5%6^78*9(0qQwWeErtTyYuiIoOpPasSdDfgGhHjJklLzZxcCvVbBnm"
    SYMBOL_MAP = {'!':'1', '@':'2', '#':'3', '$':'4', '%':'5', '^':'6', '&':'7', '*':'8', '(':'9', ')':'0'}
    def __init__(self):
        c4_index = self.LAYOUT.find('t')
        self.base_note = 60 - c4_index
        self.key_map = {self.base_note + i: key for i, key in enumerate(self.LAYOUT)}
        self.min_pitch, self.max_pitch = self.base_note, self.base_note + len(self.LAYOUT) - 1
    def get_key_for_pitch(self, pitch: int) -> Optional[str]:
        if self.min_pitch <= pitch <= self.max_pitch: return self.key_map.get(pitch)
        transposed_pitch = pitch
        if transposed_pitch < self.min_pitch:
            while transposed_pitch < self.min_pitch: transposed_pitch += 12
        elif transposed_pitch > self.max_pitch:
            while transposed_pitch > self.max_pitch: transposed_pitch -= 12
        return self.key_map.get(transposed_pitch)
    def get_key_press_info(self, key_char: str) -> Tuple[List[Key], str]:
        if key_char in self.SYMBOL_MAP: return ([Key.shift], self.SYMBOL_MAP[key_char])
        if key_char.isupper(): return ([Key.shift], key_char.lower())
        return ([], key_char)

def get_time_groups(notes: List[Note], threshold: float = 0.015) -> List[List[Note]]:
    if not notes: return []
    groups, current_group = [], [notes[0]]
    for i in range(1, len(notes)):
        if notes[i].start_time - current_group[0].start_time <= threshold: current_group.append(notes[i])
        else:
            groups.append(current_group)
            current_group = [notes[i]]
    groups.append(current_group)
    return groups

class Humanizer:
    """Applies human-like variations with stateful timing drift and decay."""
    def __init__(self, config: argparse.Namespace):
        self.config = config
        self.left_hand_drift = 0.0
        self.right_hand_drift = 0.0

    def apply_to_hand(self, notes: List[Note], hand: str, resync_points: Set[float]):
        if not any([self.config.vary_timing, self.config.vary_articulation, self.config.vary_velocity, self.config.enable_drift_correction, self.config.enable_chord_roll]):
            return
        
        time_groups = get_time_groups(notes)
        for group in time_groups:
            is_resync_point = round(group[0].start_time, 2) in resync_points
            if self.config.enable_drift_correction and is_resync_point:
                if hand == 'left': self.left_hand_drift *= self.config.drift_decay_factor
                else: self.right_hand_drift *= self.config.drift_decay_factor

            group_timing_offset = 0.0
            if self.config.vary_timing:
                group_timing_offset = (random.random() - 0.5) * 2 * self.config.timing_variance

            group_articulation = self.config.articulation
            if self.config.vary_articulation:
                group_articulation -= (random.random() * 0.1)

            if self.config.enable_chord_roll and not self.config.no_chord_roll and len(group) > 1:
                group.sort(key=lambda n: n.pitch)
                for i, note in enumerate(group):
                    note.start_time += i * 0.006

            for note in group:
                current_drift = self.left_hand_drift if hand == 'left' else self.right_hand_drift
                
                note.start_time += group_timing_offset
                if self.config.enable_drift_correction:
                    note.start_time += current_drift

                note.duration *= group_articulation

                if self.config.vary_velocity:
                    note.velocity = max(1, int(note.velocity * (1 + (random.random() - 0.5) * 0.2)))

            if self.config.enable_drift_correction:
                if hand == 'left': self.left_hand_drift += group_timing_offset
                else: self.right_hand_drift += group_timing_offset

class FingeringEngine:
    """
    Assigns hands to notes based on a biomechanical cost model, simulating
    the decisions of a human pianist. This replaces simple pitch-splitting.
    """
    TRAVEL_WEIGHT, RECENCY_WEIGHT, STRETCH_WEIGHT = 1.0, 150.0, 0.5
    CROSSOVER_PENALTY, THUMB_ON_BLACK_KEY_PENALTY = 50.0, 20.0
    MAX_HAND_SPAN = 14
    BLACK_KEY_PITCH_CLASSES = {1, 3, 6, 8, 10}

    def __init__(self):
        self.fingers = [Finger(id=i, hand='left') for i in range(5)] + [Finger(id=i, hand='right') for i in range(5, 10)]

    def assign_hands(self, notes: List[Note]):
        time_groups = get_time_groups(notes)
        for group in time_groups:
            if len(group) == 1: self._assign_single_note(group[0])
            else: self._assign_chord(group)
    
    def _update_finger_state(self, finger: Finger, note: Note):
        finger.current_pitch = note.pitch
        finger.last_press_time = note.start_time

    def _calculate_cost(self, finger: Finger, note: Note) -> float:
        if finger.current_pitch is None: return 0
        travel_cost = abs(finger.current_pitch - note.pitch) * self.TRAVEL_WEIGHT
        recency_cost = 0
        if finger.id in [f.id for f in self.fingers if f.last_press_time == finger.last_press_time]:
             time_gap = note.start_time - finger.last_press_time
             if 1e-6 < time_gap < 0.5: recency_cost = self.RECENCY_WEIGHT / time_gap
        thumb_cost = self.THUMB_ON_BLACK_KEY_PENALTY if finger.id in [0, 5] and (note.pitch % 12) in self.BLACK_KEY_PITCH_CLASSES else 0
        stretch_cost, crossover_cost = 0, 0
        other_fingers_on_hand = [f for f in self.fingers if f.hand == finger.hand and f.id != finger.id and f.current_pitch is not None]
        if other_fingers_on_hand:
            all_pitches = [f.current_pitch for f in other_fingers_on_hand] + [note.pitch]
            span = max(all_pitches) - min(all_pitches)
            if span > self.MAX_HAND_SPAN: stretch_cost = (span - self.MAX_HAND_SPAN) * self.STRETCH_WEIGHT
            for other in other_fingers_on_hand:
                if (finger.id > other.id and note.pitch < other.current_pitch) or (finger.id < other.id and note.pitch > other.current_pitch):
                    crossover_cost = self.CROSSOVER_PENALTY
                    break
        return travel_cost + recency_cost + thumb_cost + stretch_cost + crossover_cost

    def _assign_single_note(self, note: Note):
        costs = [(self._calculate_cost(f, note), f) for f in self.fingers]
        _, best_finger = min(costs, key=lambda x: x[0])
        note.hand = best_finger.hand
        self._update_finger_state(best_finger, note)

    def _assign_chord(self, chord_notes: List[Note]):
        chord_notes.sort(key=lambda n: n.pitch)
        span = chord_notes[-1].pitch - chord_notes[0].pitch
        if span > self.MAX_HAND_SPAN:
            best_gap, split_index = -1, -1
            for i in range(len(chord_notes) - 1):
                gap = chord_notes[i+1].pitch - chord_notes[i].pitch
                if gap > best_gap: best_gap, split_index = gap, i + 1
            for i, note in enumerate(chord_notes): note.hand = 'left' if i < split_index else 'right'
        else:
            left_fingers = [f for f in self.fingers if f.hand == 'left' and f.current_pitch is not None]
            right_fingers = [f for f in self.fingers if f.hand == 'right' and f.current_pitch is not None]
            left_center = np.mean([f.current_pitch for f in left_fingers]) if left_fingers else 48
            right_center = np.mean([f.current_pitch for f in right_fingers]) if right_fingers else 72
            chord_center = np.mean([n.pitch for n in chord_notes])
            chosen_hand = 'left' if abs(chord_center - left_center) <= abs(chord_center - right_center) else 'right'
            for note in chord_notes: note.hand = chosen_hand
        for note in chord_notes:
            hand_fingers = [f for f in self.fingers if f.hand == note.hand]
            for f in hand_fingers: f.last_press_time = note.start_time
            closest_finger = min(hand_fingers, key=lambda f: abs(f.current_pitch - note.pitch) if f.current_pitch else float('inf'))
            closest_finger.current_pitch = note.pitch

class PedalGenerator:
    """Generates pedal events based on a chosen musical style and hierarchical analysis."""
    @staticmethod
    def _apply_clarity_legato(events: List[KeyEvent], time_groups: List[List[Note]]):
        for i in range(len(time_groups)):
            press_time = time_groups[i][0].start_time
            release_time = time_groups[i+1][0].start_time if i < len(time_groups) - 1 else max(n.start_time + n.duration for n in time_groups[i])
            if release_time > press_time:
                events.append(KeyEvent(press_time, 1, 'pedal', 'down'))
                events.append(KeyEvent(release_time, 3, 'pedal', 'up'))

    @staticmethod
    def _apply_accent_pedal(events: List[KeyEvent], time_groups: List[List[Note]]):
        for group in time_groups:
            press_time = group[0].start_time
            release_time = max(n.start_time + n.duration for n in group)
            if release_time > press_time:
                events.append(KeyEvent(press_time, 1, 'pedal', 'down'))
                events.append(KeyEvent(release_time, 3, 'pedal', 'up'))
    
    @staticmethod
    def _apply_bridge_pedal(events: List[KeyEvent], time_groups: List[List[Note]]):
        if not time_groups: return
        press_time = time_groups[0][0].start_time
        release_time = max(n.start_time + n.duration for n in time_groups[-1])
        if release_time > press_time:
            events.append(KeyEvent(press_time, 1, 'pedal', 'down'))
            events.append(KeyEvent(release_time, 3, 'pedal', 'up'))

    @staticmethod
    def generate_events(config: argparse.Namespace, final_notes: List[Note], sections: List[MusicalSection]) -> List[KeyEvent]:
        style = config.pedal_style
        if style == 'none' or not final_notes: return []
        events = []
        for sec in sections:
            if sec.is_bridge:
                sec_final_notes = [n for n in final_notes if sec.start_time <= n.start_time < sec.end_time]
                PedalGenerator._apply_bridge_pedal(events, get_time_groups(sec_final_notes))
                continue
            for phrase in sec.rhythmic_phrases:
                phrase_final_notes = [n for n in final_notes if phrase.start_time <= n.start_time < phrase.end_time]
                phrase_time_groups = get_time_groups(phrase_final_notes)
                if not phrase_time_groups: continue
                if phrase.pattern_label == 'arpeggio':
                    PedalGenerator._apply_bridge_pedal(events, phrase_time_groups)
                    continue
                elif phrase.pattern_label in ['scale', 'ornament']:
                    continue
                if style == 'hybrid':
                    if phrase.articulation_label in ['staccato', 'staccatissimo', 'tenuto']: PedalGenerator._apply_accent_pedal(events, phrase_time_groups)
                    else: PedalGenerator._apply_clarity_legato(events, phrase_time_groups)
                elif style == 'legato':
                    if phrase.articulation_label in ['legato', 'tenuto', 'uniform']: PedalGenerator._apply_clarity_legato(events, phrase_time_groups)
                elif style == 'rhythmic':
                    if phrase.articulation_label in ['staccato', 'staccatissimo', 'tenuto']: PedalGenerator._apply_accent_pedal(events, phrase_time_groups)
        return events

class SectionAnalyzer:
    """Performs hierarchical analysis of musical structure, including pattern recognition."""
    ARTICULATION_LABELS = {
        1: ['uniform'], 2: ['legato', 'staccato'], 3: ['legato', 'tenuto', 'staccato'], 4: ['legato', 'tenuto', 'staccato', 'staccatissimo']
    }

    def __init__(self, notes: List[Note], tempo_map: TempoMap):
        self.notes, self.tempo_map = notes, tempo_map
        self.time_groups = get_time_groups(notes)

    def analyze(self) -> List[MusicalSection]:
        if not self.time_groups: return []
        phrase_boundaries = self._plan_phrases_with_bridges()
        if not phrase_boundaries: return []
        initial_phrases = self._create_sections_from_boundaries(phrase_boundaries)
        merged_sections = self._merge_similar_sections(initial_phrases)
        classified_sections = self._classify_sections_by_pace(merged_sections)
        for section in classified_sections:
            if section.is_bridge:
                self._finalize_rhythmic_phrase(section, get_time_groups(section.notes), 'bridge_held', 'standard')
                continue
            self._subdivide_section_by_articulation(section)
        return classified_sections

    def _check_for_bridge(self, group1: List[Note], group2: List[Note]) -> bool:
        gap_start = max(n.start_time + n.duration for n in group1)
        gap_end = group2[0].start_time
        gap_duration = gap_end - gap_start
        if gap_duration <= 0.1: return False
        tempo = self.tempo_map.get_tempo_at(gap_start)
        beat_duration = tempo / 1_000_000.0
        normalized_gap = gap_duration / beat_duration if beat_duration > 0 else 0
        if not (0 < normalized_gap < 1.5): return False
        hand1 = max(set(n.hand for n in group1), key=list(n.hand for n in group1).count)
        hand2 = max(set(n.hand for n in group2), key=list(n.hand for n in group2).count)
        if hand1 != hand2: return False
        avg_pitch1 = np.mean([n.pitch for n in group1])
        avg_pitch2 = np.mean([n.pitch for n in group2])
        if abs(avg_pitch1 - avg_pitch2) >= 12: return False
        return True

    def _plan_phrases_with_bridges(self) -> List[Tuple[int, int, bool]]:
        plan = []
        if not self.time_groups: return plan
        current_phrase_start_index, i = 0, 0
        while i < len(self.time_groups) - 1:
            is_bridge = self._check_for_bridge(self.time_groups[i], self.time_groups[i+1])
            if not is_bridge:
                plan.append((current_phrase_start_index, i, False))
                current_phrase_start_index = i + 1
            i += 1
        plan.append((current_phrase_start_index, len(self.time_groups) - 1, False))
        return plan

    def _create_sections_from_boundaries(self, boundaries: List[Tuple[int, int, bool]]) -> List[MusicalSection]:
        sections = []
        for start_idx, end_idx, is_bridge in boundaries:
            section_notes = [note for i in range(start_idx, end_idx + 1) for note in self.time_groups[i]]
            if not section_notes: continue
            start_time, end_time = self.time_groups[start_idx][0].start_time, max(n.start_time + n.duration for n in self.time_groups[end_idx])
            note_count, musical_beats = len(section_notes), self._calculate_musical_beats(start_time, end_time, self.tempo_map)
            normalized_density = note_count / musical_beats if musical_beats > 0 else 0
            section = MusicalSection(start_time, end_time, note_count, section_notes, normalized_density, is_bridge=is_bridge)
            sections.append(section)
        return sections

    def _merge_similar_sections(self, sections: List[MusicalSection], similarity_threshold: float = 0.35) -> List[MusicalSection]:
        if not sections: return []
        merged, current_section = [], copy.deepcopy(sections[0])
        for next_section in sections[1:]:
            if current_section.is_bridge or next_section.is_bridge:
                merged.append(current_section)
                current_section = copy.deepcopy(next_section)
                continue
            density_diff = abs(next_section.normalized_density - current_section.normalized_density)
            if density_diff / max(current_section.normalized_density, 1e-6) < similarity_threshold:
                current_section.end_time = next_section.end_time
                current_section.notes.extend(next_section.notes)
                current_section.note_count = len(current_section.notes)
                musical_beats = self._calculate_musical_beats(current_section.start_time, current_section.end_time, self.tempo_map)
                current_section.normalized_density = current_section.note_count / musical_beats if musical_beats > 0 else 0
            else:
                merged.append(current_section)
                current_section = copy.deepcopy(next_section)
        merged.append(current_section)
        return merged

    def _classify_sections_by_pace(self, sections: List[MusicalSection]) -> List[MusicalSection]:
        if not sections or len(sections) < 3:
            for section in sections: section.pace_label = 'normal'
            return sections
        densities = [s.normalized_density for s in sections]
        slow_q, fast_q = np.percentile(densities, [33, 67])
        for section in sections:
            if section.normalized_density <= slow_q: section.pace_label = 'slow'
            elif section.normalized_density >= fast_q: section.pace_label = 'fast'
            else: section.pace_label = 'normal'
        return sections
    
    def _subdivide_section_by_articulation(self, section: MusicalSection):
        time_groups = get_time_groups(section.notes)
        if len(time_groups) < 3:
            self._finalize_rhythmic_phrase(section, time_groups, 'uniform', self._detect_pattern(section.notes))
            return
        inter_chord_gaps = []
        for i in range(len(time_groups) - 1):
            gap_start = max(n.start_time + n.duration for n in time_groups[i])
            gap_end = time_groups[i+1][0].start_time
            tempo = self.tempo_map.get_tempo_at(gap_start)
            beat_duration = tempo / 1_000_000.0
            normalized_gap = (gap_end - gap_start) / beat_duration if beat_duration > 0 else 0
            inter_chord_gaps.append(max(0, normalized_gap))
        gap_data = np.array(inter_chord_gaps).reshape(-1, 1)
        inertias, max_k = [], min(len(np.unique(gap_data)), 4)
        if max_k < 2:
            self._finalize_rhythmic_phrase(section, time_groups, 'uniform', self._detect_pattern(section.notes))
            return
        for k in range(1, max_k + 1):
            kmeans = KMeans(n_clusters=k, n_init='auto', random_state=0).fit(gap_data)
            inertias.append(kmeans.inertia_)
        optimal_k = 1
        if len(inertias) > 1:
            diffs = np.diff(inertias)
            if len(diffs) > 1: optimal_k = np.argmax(np.diff(diffs)) + 2
            else: optimal_k = 2
        kmeans = KMeans(n_clusters=optimal_k, n_init='auto', random_state=0).fit(gap_data)
        sorted_indices = np.argsort(kmeans.cluster_centers_.flatten())
        labels_for_k = self.ARTICULATION_LABELS.get(optimal_k, self.ARTICULATION_LABELS[4])
        label_map = {idx: label for idx, label in zip(sorted_indices, labels_for_k)}
        gap_labels = [label_map[label] for label in kmeans.labels_]
        current_phrase_groups, current_label = [time_groups[0]], gap_labels[0]
        for i, label in enumerate(gap_labels):
            if label == current_label: current_phrase_groups.append(time_groups[i+1])
            else:
                phrase_notes = [n for g in current_phrase_groups for n in g]
                self._finalize_rhythmic_phrase(section, current_phrase_groups, current_label, self._detect_pattern(phrase_notes))
                current_phrase_groups, current_label = [time_groups[i+1]], label
        phrase_notes = [n for g in current_phrase_groups for n in g]
        self._finalize_rhythmic_phrase(section, current_phrase_groups, current_label, self._detect_pattern(phrase_notes))

    @staticmethod
    def _finalize_rhythmic_phrase(section: MusicalSection, phrase_groups: List[List[Note]], articulation: str, pattern: str):
        if not phrase_groups: return
        start_time, end_time = phrase_groups[0][0].start_time, max(n.start_time + n.duration for n in phrase_groups[-1])
        notes = [note for group in phrase_groups for note in group]
        section.rhythmic_phrases.append(RhythmicPhrase(start_time, end_time, notes, articulation, pattern))
    
    @staticmethod
    def _calculate_musical_beats(start_time: float, end_time: float, tempo_map: TempoMap) -> float:
        if start_time >= end_time: return 0.0
        total_beats = 0.0
        change_points = [event_time for event_time, _ in tempo_map.events if start_time < event_time < end_time]
        all_points = sorted(list(set([start_time] + change_points + [end_time])))
        for i in range(len(all_points) - 1):
            seg_start, seg_end = all_points[i], all_points[i+1]
            seg_duration = seg_end - seg_start
            tempo = tempo_map.get_tempo_at(seg_start)
            beat_duration = tempo / 1_000_000.0
            if beat_duration > 0: total_beats += seg_duration / beat_duration
        return total_beats

    @staticmethod
    def _detect_pattern(phrase_notes: List[Note]) -> str:
        note_count = len(phrase_notes)
        if note_count < 4: return 'standard'
        pitch_classes, pitches = {n.pitch % 12 for n in phrase_notes}, [n.pitch for n in phrase_notes]
        if len(pitch_classes) <= 4:
            root = min(pitch_classes)
            intervals = sorted([(p - root) % 12 for p in pitch_classes])
            if set(intervals).issubset({0, 4, 7, 11}) or set(intervals).issubset({0, 3, 7, 10}): return 'arpeggio'
        deltas = np.abs(np.diff(pitches))
        if len(deltas) > 0 and np.sum((deltas > 0) & (deltas <= 2)) / len(deltas) > 0.8: return 'scale'
        duration = phrase_notes[-1].start_time - phrase_notes[0].start_time
        if duration > 0 and note_count / duration > 15 and len(set(pitches)) <= 3: return 'ornament'
        return 'standard'

class Player:
    """Orchestrates parsing, humanization, and keyboard control."""
    def __init__(self, config: argparse.Namespace):
        self.config = config
        self.keyboard = Controller()
        self.parser = MidiParser()
        self.mapper = KeyMapper()
        self.humanizer = Humanizer(config)
        self.pedal_generator = PedalGenerator()
        self.event_queue: List[KeyEvent] = []
        self.stop_event = threading.Event()
        self.key_states: Dict[str, KeyState] = {}
        self.pedal_is_down = False
        self.scheduler_thread = None
        self.tempo_map = None

    def _initialize_and_analyze(self) -> Tuple[List[Note], List[MusicalSection]]:
        print(f"Loading '{self.config.midi_file}'...")
        original_notes, tempo_data = self.parser.parse(self.config.midi_file, self.config.tempo, self.config.debug)
        if not original_notes: return None, None

        if self.config.piano_only:
            print("Using advanced piano fingering model for hand assignment.")
            engine = FingeringEngine()
            engine.assign_hands(original_notes)
        else:
            print("Using simple pitch-split for hand assignment.")
            self._separate_hands_by_pitch(original_notes)
            
        self.tempo_map = TempoMap(tempo_data)
        print(f"Loaded {len(original_notes)} notes. Estimated duration: {max(n.start_time + n.duration for n in original_notes):.2f}s")
        print("\nAnalyzing musical structure...")
        analyzer = SectionAnalyzer(original_notes, self.tempo_map)
        sections = analyzer.analyze()
        if self.config.debug:
            print(f"Found {len(sections)} major sections:")
            for i, sec in enumerate(sections):
                bridge_info = "(BRIDGE)" if sec.is_bridge else ""
                print(f"  - Section {i+1} ({sec.pace_label.upper()}) {bridge_info}: {sec.start_time:.2f}s - {sec.end_time:.2f}s")
                for j, phrase in enumerate(sec.rhythmic_phrases):
                    print(f"    > Sub-Phrase {j+1} (Artic: {phrase.articulation_label}, Pattern: {phrase.pattern_label})")
            print("-" * 20)
        return original_notes, sections

    def play(self):
        self._setup_signal_handler()
        try:
            original_notes, sections = self._initialize_and_analyze()
            if not original_notes:
                print("No playable notes found.")
                return
            humanized_notes = copy.deepcopy(original_notes)
            
            left_hand_notes = [n for n in humanized_notes if n.hand == 'left']
            right_hand_notes = [n for n in humanized_notes if n.hand == 'right']
            resync_points = {round(n.start_time, 2) for n in left_hand_notes}.intersection({round(n.start_time, 2) for n in right_hand_notes})

            self.humanizer.apply_to_hand(left_hand_notes, 'left', resync_points)
            self.humanizer.apply_to_hand(right_hand_notes, 'right', resync_points)
            
            all_notes = left_hand_notes + right_hand_notes
            all_notes.sort(key=lambda n: n.start_time)

            self._schedule_events(all_notes, sections)
            if self.config.countdown: self._run_countdown()
            if self.stop_event.is_set(): return
            self.scheduler_thread = threading.Thread(target=self._run_scheduler)
            self.scheduler_thread.start()
            self.scheduler_thread.join()
        except (IOError, ValueError) as e:
            print(f"Error: {e}")
        finally:
            self.shutdown()
            
    def _separate_hands_by_pitch(self, notes: List[Note], split_point: int = 60):
        for note in notes:
            note.hand = 'left' if note.pitch < split_point else 'right'
        
    def _run_countdown(self):
        print("Get ready...")
        for i in range(3, 0, -1):
            if self.stop_event.is_set(): return
            print(f"{i}...", end=' ', flush=True)
            time.sleep(1)
        print("Playing!")

    def _schedule_events(self, notes_to_play: List[Note], sections: List[MusicalSection]):
        self.key_states.clear()
        for note in notes_to_play:
            key_char = self.mapper.get_key_for_pitch(note.pitch)
            if key_char:
                heapq.heappush(self.event_queue, KeyEvent(note.start_time, 2, 'press', key_char))
                heapq.heappush(self.event_queue, KeyEvent(note.start_time + note.duration, 4, 'release', key_char))
                if key_char not in self.key_states:
                    self.key_states[key_char] = KeyState(key_char)
        pedal_events = self.pedal_generator.generate_events(self.config, notes_to_play, sections)
        for event in pedal_events:
            heapq.heappush(self.event_queue, event)

    def _run_scheduler(self):
        if not self.event_queue: return
        start_time = time.perf_counter()
        while not self.stop_event.is_set() and self.event_queue:
            playback_time = time.perf_counter() - start_time
            if self.event_queue[0].time <= playback_time:
                event = heapq.heappop(self.event_queue)
                self._execute_key_event(event)
            else:
                time.sleep(0.001)

    def _physical_press(self, modifiers: List[Key], base_key: str):
        if Key.shift in modifiers:
            with self.keyboard.pressed(Key.shift): self.keyboard.press(base_key)
        else: self.keyboard.press(base_key)
            
    def _execute_key_event(self, event: KeyEvent):
        if event.action == 'pedal':
            self._handle_pedal_event(event)
            return
        key_char = event.key_char
        state = self.key_states.get(key_char)
        if not state: return
        modifiers, base_key = self.mapper.get_key_press_info(key_char)
        try:
            if event.action == 'press':
                was_physically_down = state.is_physically_down
                is_sustained_only = state.is_sustained and not state.is_active
                state.press()
                if is_sustained_only:
                    self.keyboard.release(base_key)
                    self._physical_press(modifiers, base_key)
                elif not was_physically_down:
                    self._physical_press(modifiers, base_key)
            elif event.action == 'release':
                was_physically_down = state.is_physically_down
                state.release(self.pedal_is_down)
                if was_physically_down and not state.is_physically_down:
                    self.keyboard.release(base_key)
        except Exception: pass
            
    def _handle_pedal_event(self, event: KeyEvent):
        if event.key_char == 'down' and not self.pedal_is_down:
            self.pedal_is_down = True
            try: self.keyboard.press(Key.space)
            except Exception: pass
        elif event.key_char == 'up' and self.pedal_is_down:
            self.pedal_is_down = False
            try: self.keyboard.release(Key.space)
            except Exception: pass
            for key_char, state in self.key_states.items():
                was_physically_down = state.is_physically_down
                state.lift_sustain()
                if was_physically_down and not state.is_physically_down:
                    try:
                        _, base_key = self.mapper.get_key_press_info(key_char)
                        self.keyboard.release(base_key)
                    except Exception: pass

    def _signal_handler(self, signum, frame):
        if not self.stop_event.is_set():
            print("\nPlayback interrupted by user.")
            self.stop_event.set()

    def _setup_signal_handler(self):
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def shutdown(self):
        self.stop_event.set()
        if self.scheduler_thread and threading.current_thread() != self.scheduler_thread:
            self.scheduler_thread.join(timeout=1)
        print("Shutting down and releasing all keys.")
        # FIX: Corrected typo from self.key_stats to self.key_states
        for key_char, state in self.key_states.items():
            if state.is_physically_down:
                try:
                    _, base_key = self.mapper.get_key_press_info(key_char)
                    self.keyboard.release(base_key)
                except Exception: pass
        self.key_states.clear()
        if self.pedal_is_down:
            try: self.keyboard.release(Key.space)
            except Exception: pass
        for key in [Key.shift, Key.ctrl, Key.alt]:
            try: self.keyboard.release(key)
            except Exception: pass

def main():
    parser = argparse.ArgumentParser(
        description="An intelligent MIDI-to-Keyboard performance engine.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    playback_group = parser.add_argument_group('Playback Controls')
    playback_group.add_argument('midi_file', help='Path to the MIDI file to play.')
    playback_group.add_argument('--tempo', type=float, default=1.0, 
                                help='Adjust playback speed. Example: 0.5 for half-speed, 2.0 for double-speed.')
    playback_group.add_argument('--no-countdown', dest='countdown', action='store_false', 
                                help='Skip the 3-second countdown before playback starts.')
    playback_group.add_argument('--debug', action='store_true', 
                                help='Print detailed analysis reports and debug information during playback.')

    analysis_group = parser.add_argument_group('Analysis & Interpretation')
    analysis_group.add_argument('--piano-only', action='store_true', 
                                help='Use an advanced biomechanical model for hand assignment. Recommended for piano-specific MIDI files.')
    analysis_group.add_argument('--pedal-style', choices=['none', 'legato', 'rhythmic', 'hybrid'], default='hybrid', 
                                help="'hybrid': Balances clarity and connection (recommended). 'legato': Prioritizes connecting notes. 'rhythmic': Accents strong beats. 'none': Disables pedal.")

    human_group = parser.add_argument_group('Humanization Controls (for a more natural performance)')
    human_group.add_argument('-n', '--natural', action='store_true', 
                                help='Convenience flag to enable all core humanization options with sensible defaults.')
    human_group.add_argument('--vary-timing', action='store_true', 
                                help='Add slight, human-like random shifts to note start times.')
    human_group.add_argument('--vary-articulation', action='store_true', 
                                help='Add slight, human-like random changes to note durations (how long a note is held).')
    human_group.add_argument('--vary-velocity', action='store_true',
                                help="Add slight, human-like random variations to note velocity (how 'hard' a note is played).")
    human_group.add_argument('--enable-chord-roll', action='store_true', 
                                help='Simulate rolling chords by slightly offsetting the start times of notes within them.')
    human_group.add_argument('--no-chord-roll', action='store_true',
                                help='Force all notes in a chord to be played at the exact same time. Overrides --enable-chord-roll.')
    human_group.add_argument('--enable-drift-correction', action='store_true', 
                                help="Simulate how a pianist's hands can drift apart in timing and then resynchronize.")
    human_group.add_argument('--timing-variance', type=float, default=0.01, 
                                help='Controls the magnitude of timing shifts (larger value = more variation).')
    human_group.add_argument('--articulation', type=float, default=0.95, 
                                help='Base multiplier for note duration. <1.0 is more staccato, >1.0 is more legato.')
    human_group.add_argument('--drift-decay-factor', type=float, default=0.25,
                                help='How quickly hands resynchronize at shared chords (0=instantly, 1=never).')
    
    args = parser.parse_args()
    if args.natural:
        args.vary_timing = True
        args.vary_articulation = True
        args.vary_velocity = True
        args.enable_drift_correction = True
        if not args.no_chord_roll:
            args.enable_chord_roll = True
        
    player = Player(args)
    player.play()

if __name__ == "__main__":
    main()