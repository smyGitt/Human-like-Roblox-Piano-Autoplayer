#!/usr/bin/env python3
#
# MIDI2Key: A robust MIDI performance engine with a user-friendly GUI.
# Final version incorporating functional logic and all requested features.
#
import mido
import time
import heapq
import threading
import random
import copy
import numpy as np
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
import os
import sys
from PyQt6.QtGui import QIcon

# --- GUI Dependencies ---
try:
    from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                                 QPushButton, QCheckBox, QSlider, QLabel, QFileDialog,
                                 QGroupBox, QFormLayout, QTabWidget, QTextEdit, QProgressBar,
                                 QComboBox, QDoubleSpinBox, QMessageBox, QSpinBox, QGridLayout)
    from PyQt6.QtCore import QObject, QThread, pyqtSignal as Signal, Qt
    from PyQt6.QtGui import QFont
except ImportError:
    print("PyQt6 not found. Please run 'pip install PyQt6' to run the GUI.")
    sys.exit(1)

# --- Engine Dependencies ---
try:
    from sklearn.cluster import KMeans
except ImportError:
    print("Scikit-learn not found. Please run 'pip install scikit-learn' for advanced pedal analysis.")
    sys.exit(1)

try:
    from pynput.keyboard import Key, Controller
except ImportError:
    print("pynput not found. Please run 'pip install pynput'.")
    sys.exit(1)


# =====================================================================================
# ==                                                                                 ==
# ==                       SECTION 1: CORE ENGINE & DATA STRUCTURES                  ==
# ==                                                                                 ==
# =====================================================================================

@dataclass
class Note:
    """Represents a single, parsed musical note."""
    id: int # Unique ID for tracking
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
    is_active: bool = False
    is_sustained: bool = False

    def press(self):
        self.is_active = True

    def release(self, pedal_is_down: bool):
        self.is_active = False
        self.is_sustained = True if pedal_is_down else False

    def lift_sustain(self):
        self.is_sustained = False

    @property
    def is_physically_down(self) -> bool:
        return self.is_active or self.is_sustained

@dataclass
class Finger:
    """Represents the state of a single physical finger."""
    id: int
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


class MidiParser:
    @staticmethod
    def parse(filepath: str, tempo_scale: float = 1.0) -> Tuple[List[Note], List[Tuple[float, int]]]:
        try:
            mid = mido.MidiFile(filepath)
        except Exception as e:
            raise IOError(f"Could not read or parse MIDI file: {e}")
        notes: List[Note] = []
        tempo_map_data: List[Tuple[float, int]] = []
        open_notes: Dict[int, List[Dict]] = defaultdict(list)
        absolute_time: float = 0.0
        tempo = 500000
        ticks_per_beat = mid.ticks_per_beat or 480
        tempo_map_data.append((0.0, tempo))
        note_id_counter = 0
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
                        scaled_start = start / tempo_scale
                        scaled_duration = duration / tempo_scale
                        notes.append(Note(id=note_id_counter, pitch=msg.note, velocity=note_data['vel'], start_time=scaled_start, duration=scaled_duration))
                        note_id_counter += 1
        notes.sort(key=lambda n: n.start_time)
        return notes, tempo_map_data

class KeyMapper:
    LAYOUT = "1!2@34$5%6^78*9(0qQwWeErtTyYuiIoOpPasSdDfgGhHjJklLzZxcCvVbBnm"
    SYMBOL_MAP = {'!': '1', '@': '2', '#': '3', '$': '4', '%': '5', '^': '6', '&': '7', '*': '8', '(': '9', ')': '0'}
    PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    BLACK_KEY_PITCH_CLASSES = {1, 3, 6, 8, 10}
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
    @staticmethod
    def pitch_to_name(pitch: int) -> str:
        octave = (pitch // 12) - 1
        note_name = KeyMapper.PITCH_NAMES[pitch % 12]
        return f"{note_name}{octave}"
    @staticmethod
    def is_black_key(pitch: int) -> bool:
        return pitch % 12 in KeyMapper.BLACK_KEY_PITCH_CLASSES

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
    def __init__(self, config: Dict):
        self.config = config
        self.left_hand_drift = 0.0
        self.right_hand_drift = 0.0

    def apply_to_hand(self, notes: List[Note], hand: str, resync_points: Set[float]):
        if not any([self.config.get('vary_timing'), self.config.get('vary_articulation'), self.config.get('vary_velocity'), self.config.get('enable_drift_correction'), self.config.get('enable_chord_roll')]):
            return
        time_groups = get_time_groups(notes)
        for group in time_groups:
            is_resync_point = round(group[0].start_time, 2) in resync_points
            if self.config.get('enable_drift_correction') and is_resync_point:
                if hand == 'left': self.left_hand_drift *= self.config.get('drift_decay_factor')
                else: self.right_hand_drift *= self.config.get('drift_decay_factor')
            group_timing_offset = 0.0
            if self.config.get('vary_timing'):
                group_timing_offset = (random.random() - 0.5) * 2 * self.config.get('timing_variance')
            group_articulation = self.config.get('articulation')
            if self.config.get('vary_articulation'):
                group_articulation -= (random.random() * 0.1)
            if self.config.get('enable_chord_roll') and len(group) > 1:
                group.sort(key=lambda n: n.pitch)
                for i, note in enumerate(group): note.start_time += i * 0.006
            for note in group:
                current_drift = self.left_hand_drift if hand == 'left' else self.right_hand_drift
                note.start_time += group_timing_offset
                if self.config.get('enable_drift_correction'):
                    note.start_time += current_drift
                note.duration *= group_articulation
                if self.config.get('vary_velocity'):
                    note.velocity = max(1, int(note.velocity * (1 + (random.random() - 0.5) * 0.2)))
            if self.config.get('enable_drift_correction'):
                if hand == 'left': self.left_hand_drift += group_timing_offset
                else: self.right_hand_drift += group_timing_offset

    def apply_tempo_rubato(self, all_notes: List[Note], sections: List[MusicalSection]):
        if not self.config.get('enable_tempo_sway'):
            return
        
        base_intensity = self.config.get('tempo_sway_intensity', 0.0)
        note_map = {note.id: note for note in all_notes}

        for section in sections:
            if section.pace_label == 'fast':
                pace_multiplier = 0.25
            elif section.pace_label == 'slow':
                pace_multiplier = 1.5
            else:
                pace_multiplier = 1.0

            for phrase in section.rhythmic_phrases:
                phrase_duration = phrase.end_time - phrase.start_time
                if phrase_duration < 1.0: continue

                phrase_intensity = random.uniform(0.5, 1.0) * base_intensity * pace_multiplier
                if phrase_intensity == 0: continue

                phase_shift = random.uniform(-np.pi / 4, np.pi / 4)

                for note_in_phrase in phrase.notes:
                    if note_in_phrase.id in note_map:
                        note_to_modify = note_map[note_in_phrase.id]
                        
                        relative_pos = (note_to_modify.start_time - phrase.start_time) / phrase_duration
                        angle = relative_pos * np.pi + phase_shift
                        sine_offset = np.sin(angle)
                        
                        time_shift = sine_offset * phrase_intensity
                        note_to_modify.start_time -= time_shift


class FingeringEngine:
    TRAVEL_WEIGHT, RECENCY_WEIGHT, STRETCH_WEIGHT = 1.0, 150.0, 0.5
    CROSSOVER_PENALTY, THUMB_ON_BLACK_KEY_PENALTY = 50.0, 20.0
    MAX_HAND_SPAN = 14
    def __init__(self):
        self.fingers = [Finger(id=i, hand='left') for i in range(5)] + [Finger(id=i, hand='right') for i in range(5, 10)]
    def assign_hands(self, notes: List[Note]):
        time_groups = get_time_groups(notes)
        for group in time_groups:
            if len(group) == 1: self._assign_single_note(group[0])
            else: self._assign_chord(group)
    def _update_finger_state(self, finger: Finger, note: Note):
        finger.current_pitch = note.pitch; finger.last_press_time = note.start_time
    def _calculate_cost(self, finger: Finger, note: Note) -> float:
        if finger.current_pitch is None: return 0
        travel_cost = abs(finger.current_pitch - note.pitch) * self.TRAVEL_WEIGHT; recency_cost = 0
        if finger.id in [f.id for f in self.fingers if f.last_press_time == finger.last_press_time]:
             time_gap = note.start_time - finger.last_press_time
             if 1e-6 < time_gap < 0.5: recency_cost = self.RECENCY_WEIGHT / time_gap
        thumb_cost = self.THUMB_ON_BLACK_KEY_PENALTY if finger.id in [0, 5] and KeyMapper.is_black_key(note.pitch) else 0
        stretch_cost, crossover_cost = 0, 0
        other_fingers_on_hand = [f for f in self.fingers if f.hand == finger.hand and f.id != finger.id and f.current_pitch is not None]
        if other_fingers_on_hand:
            all_pitches = [f.current_pitch for f in other_fingers_on_hand] + [note.pitch]
            span = max(all_pitches) - min(all_pitches)
            if span > self.MAX_HAND_SPAN: stretch_cost = (span - self.MAX_HAND_SPAN) * self.STRETCH_WEIGHT
            for other in other_fingers_on_hand:
                if (finger.id > other.id and note.pitch < other.current_pitch) or (finger.id < other.id and note.pitch > other.current_pitch):
                    crossover_cost = self.CROSSOVER_PENALTY; break
        return travel_cost + recency_cost + thumb_cost + stretch_cost + crossover_cost
    def _assign_single_note(self, note: Note):
        costs = [(self._calculate_cost(f, note), f) for f in self.fingers]
        _, best_finger = min(costs, key=lambda x: x[0])
        note.hand = best_finger.hand; self._update_finger_state(best_finger, note)
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
    @staticmethod
    def _apply_clarity_legato(events: List[KeyEvent], time_groups: List[List[Note]]):
        for i in range(len(time_groups)):
            press_time = time_groups[i][0].start_time
            release_time = time_groups[i+1][0].start_time if i < len(time_groups) - 1 else max(n.start_time + n.duration for n in time_groups[i])
            if release_time > press_time:
                events.append(KeyEvent(press_time, 1, 'pedal', 'down')); events.append(KeyEvent(release_time, 3, 'pedal', 'up'))
    @staticmethod
    def _apply_accent_pedal(events: List[KeyEvent], time_groups: List[List[Note]]):
        for group in time_groups:
            press_time = group[0].start_time; release_time = max(n.start_time + n.duration for n in group)
            if release_time > press_time:
                events.append(KeyEvent(press_time, 1, 'pedal', 'down')); events.append(KeyEvent(release_time, 3, 'pedal', 'up'))
    @staticmethod
    def _apply_bridge_pedal(events: List[KeyEvent], time_groups: List[List[Note]]):
        if not time_groups: return
        press_time = time_groups[0][0].start_time; release_time = max(n.start_time + n.duration for n in time_groups[-1])
        if release_time > press_time:
            events.append(KeyEvent(press_time, 1, 'pedal', 'down')); events.append(KeyEvent(release_time, 3, 'pedal', 'up'))
    @staticmethod
    def generate_events(config: Dict, final_notes: List[Note], sections: List[MusicalSection]) -> List[KeyEvent]:
        style = config.get('pedal_style')
        if style == 'none' or not final_notes: return []
        events = []
        for sec in sections:
            if sec.is_bridge:
                sec_final_notes = [n for n in final_notes if sec.start_time <= n.start_time < sec.end_time]
                PedalGenerator._apply_bridge_pedal(events, get_time_groups(sec_final_notes)); continue
            for phrase in sec.rhythmic_phrases:
                phrase_final_notes = [n for n in final_notes if phrase.start_time <= n.start_time < phrase.end_time]
                phrase_time_groups = get_time_groups(phrase_final_notes);
                if not phrase_time_groups: continue
                if phrase.pattern_label == 'arpeggio': PedalGenerator._apply_bridge_pedal(events, phrase_time_groups); continue
                elif phrase.pattern_label in ['scale', 'ornament']: continue
                if style == 'hybrid':
                    if phrase.articulation_label in ['staccato', 'staccatissimo', 'tenuto']: PedalGenerator._apply_accent_pedal(events, phrase_time_groups)
                    else: PedalGenerator._apply_clarity_legato(events, phrase_time_groups)
                elif style == 'legato':
                    if phrase.articulation_label in ['legato', 'tenuto', 'uniform']: PedalGenerator._apply_clarity_legato(events, phrase_time_groups)
                elif style == 'rhythmic':
                    if phrase.articulation_label in ['staccato', 'staccatissimo', 'tenuto']: PedalGenerator._apply_accent_pedal(events, phrase_time_groups)
        return events

class SectionAnalyzer:
    ARTICULATION_LABELS = { 1: ['uniform'], 2: ['legato', 'staccato'], 3: ['legato', 'tenuto', 'staccato'], 4: ['legato', 'tenuto', 'staccato', 'staccatissimo'] }
    def __init__(self, notes: List[Note], tempo_map: TempoMap):
        self.notes, self.tempo_map = notes, tempo_map; self.time_groups = get_time_groups(notes)
    def analyze(self) -> List[MusicalSection]:
        if not self.time_groups: return []
        phrase_boundaries = self._plan_phrases_with_bridges();
        if not phrase_boundaries: return []
        initial_phrases = self._create_sections_from_boundaries(phrase_boundaries)
        merged_sections = self._merge_similar_sections(initial_phrases)
        classified_sections = self._classify_sections_by_pace(merged_sections)
        for section in classified_sections:
            if section.is_bridge: self._finalize_rhythmic_phrase(section, get_time_groups(section.notes), 'bridge_held', 'standard'); continue
            self._subdivide_section_by_articulation(section)
        return classified_sections
    def _check_for_bridge(self, group1: List[Note], group2: List[Note]) -> bool:
        gap_start = max(n.start_time + n.duration for n in group1); gap_end = group2[0].start_time; gap_duration = gap_end - gap_start
        if gap_duration <= 0.1: return False
        tempo = self.tempo_map.get_tempo_at(gap_start); beat_duration = tempo / 1_000_000.0
        normalized_gap = gap_duration / beat_duration if beat_duration > 0 else 0
        if not (0 < normalized_gap < 1.5): return False
        hand1 = max(set(n.hand for n in group1), key=list(n.hand for n in group1).count); hand2 = max(set(n.hand for n in group2), key=list(n.hand for n in group2).count)
        if hand1 != hand2: return False
        avg_pitch1 = np.mean([n.pitch for n in group1]); avg_pitch2 = np.mean([n.pitch for n in group2])
        if abs(avg_pitch1 - avg_pitch2) >= 12: return False
        return True
    def _plan_phrases_with_bridges(self) -> List[Tuple[int, int, bool]]:
        plan = [];
        if not self.time_groups: return plan
        current_phrase_start_index, i = 0, 0
        while i < len(self.time_groups) - 1:
            is_bridge = self._check_for_bridge(self.time_groups[i], self.time_groups[i+1])
            if not is_bridge: plan.append((current_phrase_start_index, i, False)); current_phrase_start_index = i + 1
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
            if current_section.is_bridge or next_section.is_bridge: merged.append(current_section); current_section = copy.deepcopy(next_section); continue
            density_diff = abs(next_section.normalized_density - current_section.normalized_density)
            if density_diff / max(current_section.normalized_density, 1e-6) < similarity_threshold:
                current_section.end_time = next_section.end_time; current_section.notes.extend(next_section.notes)
                current_section.note_count = len(current_section.notes)
                musical_beats = self._calculate_musical_beats(current_section.start_time, current_section.end_time, self.tempo_map)
                current_section.normalized_density = current_section.note_count / musical_beats if musical_beats > 0 else 0
            else: merged.append(current_section); current_section = copy.deepcopy(next_section)
        merged.append(current_section)
        return merged
    def _classify_sections_by_pace(self, sections: List[MusicalSection]) -> List[MusicalSection]:
        if not sections or len(sections) < 3:
            for section in sections: section.pace_label = 'normal'
            return sections
        densities = [s.normalized_density for s in sections]; slow_q, fast_q = np.percentile(densities, [33, 67])
        for section in sections:
            if section.normalized_density <= slow_q: section.pace_label = 'slow'
            elif section.normalized_density >= fast_q: section.pace_label = 'fast'
            else: section.pace_label = 'normal'
        return sections
    def _subdivide_section_by_articulation(self, section: MusicalSection):
        time_groups = get_time_groups(section.notes)
        if len(time_groups) < 3: self._finalize_rhythmic_phrase(section, time_groups, 'uniform', self._detect_pattern(section.notes)); return
        inter_chord_gaps = []
        for i in range(len(time_groups) - 1):
            gap_start = max(n.start_time + n.duration for n in time_groups[i]); gap_end = time_groups[i+1][0].start_time
            tempo = self.tempo_map.get_tempo_at(gap_start); beat_duration = tempo / 1_000_000.0
            normalized_gap = (gap_end - gap_start) / beat_duration if beat_duration > 0 else 0
            inter_chord_gaps.append(max(0, normalized_gap))
        gap_data = np.array(inter_chord_gaps).reshape(-1, 1)
        inertias, max_k = [], min(len(np.unique(gap_data)), 4)
        if max_k < 2: self._finalize_rhythmic_phrase(section, time_groups, 'uniform', self._detect_pattern(section.notes)); return
        for k in range(1, max_k + 1):
            kmeans = KMeans(n_clusters=k, n_init='auto', random_state=0).fit(gap_data); inertias.append(kmeans.inertia_)
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
            seg_start, seg_end = all_points[i], all_points[i+1]; seg_duration = seg_end - seg_start
            tempo = tempo_map.get_tempo_at(seg_start); beat_duration = tempo / 1_000_000.0
            if beat_duration > 0: total_beats += seg_duration / beat_duration
        return total_beats
    @staticmethod
    def _detect_pattern(phrase_notes: List[Note]) -> str:
        note_count = len(phrase_notes);
        if note_count < 4: return 'standard'
        pitch_classes, pitches = {n.pitch % 12 for n in phrase_notes}, [n.pitch for n in phrase_notes]
        if len(pitch_classes) <= 4:
            root = min(pitch_classes); intervals = sorted([(p - root) % 12 for p in pitch_classes])
            if set(intervals).issubset({0, 4, 7, 11}) or set(intervals).issubset({0, 3, 7, 10}): return 'arpeggio'
        deltas = np.abs(np.diff(pitches))
        if len(deltas) > 0 and np.sum((deltas > 0) & (deltas <= 2)) / len(deltas) > 0.8: return 'scale'
        duration = phrase_notes[-1].start_time - phrase_notes[0].start_time
        if duration > 0 and note_count / duration > 15 and len(set(pitches)) <= 3: return 'ornament'
        return 'standard'

# =====================================================================================
# ==                                                                                 ==
# ==           SECTION 2: FUNCTIONAL PLAYER LOGIC (INTEGRATED & ADAPTED)             ==
# ==                                                                                 ==
# =====================================================================================

class Player(QObject):
    status_updated = Signal(str)
    progress_updated = Signal(int)
    playback_finished = Signal()

    def __init__(self, config: Dict):
        super().__init__()
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
        self.tempo_map = None

    def play(self):
        try:
            original_notes, sections = self._initialize_and_analyze()
            if not original_notes:
                self.status_updated.emit("Error: No playable notes found in the file.")
                self.playback_finished.emit()
                return

            humanized_notes = copy.deepcopy(original_notes)
            left_hand_notes = [n for n in humanized_notes if n.hand == 'left']
            right_hand_notes = [n for n in humanized_notes if n.hand == 'right']
            resync_points = {round(n.start_time, 2) for n in left_hand_notes}.intersection({round(n.start_time, 2) for n in right_hand_notes})

            self.humanizer.apply_to_hand(left_hand_notes, 'left', resync_points)
            self.humanizer.apply_to_hand(right_hand_notes, 'right', resync_points)
            
            all_notes = sorted(left_hand_notes + right_hand_notes, key=lambda n: n.start_time)
            
            self.humanizer.apply_tempo_rubato(all_notes, sections)
            
            if self.config.get('debug_mode'):
                self.status_updated.emit(self._generate_debug_report(original_notes, all_notes, sections))

            self._schedule_events(all_notes, sections)

            if self.config.get('debug_mode'):
                 self.status_updated.emit(self._get_event_queue_report())

            if self.config.get('countdown'): self._run_countdown()
            if self.stop_event.is_set():
                self.playback_finished.emit()
                return

            self.status_updated.emit("Playback starting...")
            self._run_scheduler()

        except (IOError, ValueError) as e:
            self.status_updated.emit(f"Error: {e}")
        except Exception as e:
            self.status_updated.emit(f"An unexpected error occurred: {e}")
        finally:
            if not self.stop_event.is_set():
                self.shutdown()
            self.playback_finished.emit()

    def stop(self):
        if not self.stop_event.is_set():
            self.status_updated.emit("Stopping playback...")
            self.stop_event.set()
            self.shutdown()

    def _initialize_and_analyze(self) -> Tuple[Optional[List[Note]], Optional[List[MusicalSection]]]:
        self.status_updated.emit(f"Loading '{self.config.get('midi_file')}'...")
        tempo_scale = self.config.get('tempo') / 100.0
        original_notes, tempo_data = self.parser.parse(self.config.get('midi_file'), tempo_scale)
        if not original_notes: return None, None

        if self.config.get('piano_only'):
            self.status_updated.emit("Using advanced piano fingering model for hand assignment.")
            engine = FingeringEngine()
            engine.assign_hands(original_notes)
        else:
            self.status_updated.emit("Using simple pitch-split for hand assignment.")
            self._separate_hands_by_pitch(original_notes)

        self.tempo_map = TempoMap(tempo_data)
        duration = max(n.start_time + n.duration for n in original_notes) if original_notes else 0
        self.status_updated.emit(f"Loaded {len(original_notes)} notes. Estimated duration: {duration:.2f}s")

        self.status_updated.emit("Analyzing musical structure...")
        analyzer = SectionAnalyzer(original_notes, self.tempo_map)
        sections = analyzer.analyze()
        self.status_updated.emit(f"Analysis complete. Found {len(sections)} major sections.")

        return original_notes, sections

    def _generate_debug_report(self, original_notes, humanized_notes, sections) -> str:
        report = ["\n\n--- MIDI2Key Debug Report ---"]
        
        report.append("\n[1. Playback Configuration]")
        for key, val in self.config.items():
            report.append(f"  - {key}: {val}")

        report.append("\n[2. Initial Analysis Summary]")
        report.append(f"  - Total Notes Parsed: {len(original_notes)}")
        report.append(f"  - Sections Found: {len(sections)}")
        for i, sec in enumerate(sections):
            report.append(f"    - Section {i+1} ({sec.start_time:.3f}s - {sec.end_time:.3f}s): pace = {sec.pace_label}")

        report.append("\n[3. Detailed Section & Phrase Analysis]")
        for i, sec in enumerate(sections):
            report.append(f"  - SECTION {i+1} ({sec.start_time:.3f}s - {sec.end_time:.3f}s) | Pace: {sec.pace_label}")
            for j, phrase in enumerate(sec.rhythmic_phrases):
                report.append(f"    > Phrase {j+1} ({phrase.start_time:.3f}s - {phrase.end_time:.3f}s) | Artic: {phrase.articulation_label}, Pattern: {phrase.pattern_label}")
        
        report.append("\n[4. Note Transformation Log]")
        original_notes_by_id = {n.id: n for n in original_notes}
        for final_note in humanized_notes:
            original_note = original_notes_by_id.get(final_note.id)
            if original_note:
                time_delta = final_note.start_time - original_note.start_time
                dur_delta = final_note.duration - original_note.duration
                report.append(
                    f"  - Note {final_note.id:<4} ({self.mapper.pitch_to_name(final_note.pitch):<4}): "
                    f"Time: {original_note.start_time:7.3f}s -> {final_note.start_time:7.3f}s ({time_delta:+.3f}s) | "
                    f"Dur: {original_note.duration:6.3f}s -> {final_note.duration:6.3f}s ({dur_delta:+.3f}s)"
                )

        return "\n".join(report)
    
    def _get_event_queue_report(self) -> str:
        report = ["\n[5. Final Event Queue]"]
        temp_queue = sorted(list(self.event_queue))
        for event in temp_queue:
            report.append(f"  - Time: {event.time:7.3f}s | Prio: {event.priority} | Action: {event.action:<7} | Key: '{event.key_char}'")
        return "\n".join(report)

    def _separate_hands_by_pitch(self, notes: List[Note], split_point: int = 60):
        for note in notes: note.hand = 'left' if note.pitch < split_point else 'right'

    def _run_countdown(self):
        self.status_updated.emit("Get ready...")
        for i in range(3, 0, -1):
            if self.stop_event.is_set(): return
            self.status_updated.emit(f"{i}...")
            time.sleep(1)
        self.status_updated.emit("Playing!")

    def _schedule_events(self, notes_to_play: List[Note], sections: List[MusicalSection]):
        self.key_states.clear()
        
        use_mistakes = self.config.get('enable_mistakes', False)
        mistake_chance = self.config.get('mistake_chance', 0) / 100.0
        
        played_pitches_in_section = set()
        current_section_idx = -1

        for note in notes_to_play:
            note_section_idx = -1
            for i, sec in enumerate(sections):
                if sec.start_time <= note.start_time < sec.end_time:
                    note_section_idx = i
                    break
            
            if note_section_idx != current_section_idx:
                played_pitches_in_section.clear()
                current_section_idx = note_section_idx

            mistake_scheduled = False
            is_eligible_for_mistake = note.pitch not in played_pitches_in_section
            make_mistake = use_mistakes and is_eligible_for_mistake and (random.random() < mistake_chance)
            
            if make_mistake:
                mistake_duration = random.uniform(0.030, 0.060)
                mistake_pitch = self._get_mistake_pitch(note.pitch)
                
                if mistake_pitch is not None:
                    mistake_key = self.mapper.get_key_for_pitch(mistake_pitch)
                    correct_key = self.mapper.get_key_for_pitch(note.pitch)

                    if mistake_key and correct_key:
                        if note.duration > mistake_duration: # Slip-and-correct mistake
                            heapq.heappush(self.event_queue, KeyEvent(note.start_time, 2, 'press', mistake_key))
                            heapq.heappush(self.event_queue, KeyEvent(note.start_time + mistake_duration, 4, 'release', mistake_key))
                            correct_start = note.start_time + mistake_duration
                            correct_duration = note.duration - mistake_duration
                            heapq.heappush(self.event_queue, KeyEvent(correct_start, 2, 'press', correct_key))
                            heapq.heappush(self.event_queue, KeyEvent(correct_start + correct_duration, 4, 'release', correct_key))
                        else: # "Fatal" mistake for short notes
                            heapq.heappush(self.event_queue, KeyEvent(note.start_time, 2, 'press', mistake_key))
                            heapq.heappush(self.event_queue, KeyEvent(note.start_time + note.duration, 4, 'release', mistake_key))
                        
                        if correct_key not in self.key_states: self.key_states[correct_key] = KeyState(correct_key)
                        if mistake_key not in self.key_states: self.key_states[mistake_key] = KeyState(mistake_key)
                        mistake_scheduled = True

            if not mistake_scheduled:
                key_char = self.mapper.get_key_for_pitch(note.pitch)
                if key_char:
                    heapq.heappush(self.event_queue, KeyEvent(note.start_time, 2, 'press', key_char))
                    heapq.heappush(self.event_queue, KeyEvent(note.start_time + note.duration, 4, 'release', key_char))
                    if key_char not in self.key_states:
                        self.key_states[key_char] = KeyState(key_char)
            
            played_pitches_in_section.add(note.pitch)

        pedal_events = self.pedal_generator.generate_events(self.config, notes_to_play, sections)
        for event in pedal_events:
            heapq.heappush(self.event_queue, event)
            
    def _get_mistake_pitch(self, original_pitch: int) -> Optional[int]:
        is_black = KeyMapper.is_black_key(original_pitch)
        
        if is_black:
            possible_offsets = [-2, -1, 1, 2]
            offset = random.choice(possible_offsets)
            return original_pitch + offset
        else:
            valid_neighbors = []
            for offset in [-2, -1, 1, 2]:
                neighbor_pitch = original_pitch + offset
                if not KeyMapper.is_black_key(neighbor_pitch):
                    valid_neighbors.append(neighbor_pitch)
            
            if valid_neighbors:
                return random.choice(valid_neighbors)
        return None


    def _run_scheduler(self):
        if not self.event_queue: return
        start_time = time.perf_counter()
        total_duration = max(e.time for e in self.event_queue) if self.event_queue else 0

        while not self.stop_event.is_set() and self.event_queue:
            playback_time = time.perf_counter() - start_time
            if self.event_queue[0].time <= playback_time:
                event = heapq.heappop(self.event_queue)
                self._execute_key_event(event)
            else:
                time.sleep(0.001)
            
            if total_duration > 0:
                progress = int((playback_time / total_duration) * 100)
                self.progress_updated.emit(progress)

    def _physical_press(self, modifiers: List[Key], base_key: str):
        if Key.shift in modifiers:
            with self.keyboard.pressed(Key.shift): self.keyboard.press(base_key)
        else: self.keyboard.press(base_key)

    def _execute_key_event(self, event: KeyEvent):
        if self.stop_event.is_set(): return
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
        if self.stop_event.is_set(): return
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

    def shutdown(self):
        self.status_updated.emit("Releasing all keys...")
        for key_char, state in list(self.key_states.items()):
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
        self.status_updated.emit("Shutdown complete.")


# =====================================================================================
# ==                                                                                 ==
# ==                       SECTION 3: PyQt6 GUI (FINAL VERSION)                      ==
# ==                                                                                 ==
# =====================================================================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MIDI2Key")
        self.setMinimumWidth(550)
        self.player_thread = None
        self.player = None
        self.humanization_sub_checkboxes = []
        self.humanization_sliders = []
        self.humanization_spinboxes = []
        if getattr(sys, 'frozen', False):
            base_path = sys._MEIPASS
        else:
            base_path = os.path.dirname(os.path.abspath(__file__))
        
        icon_path = os.path.join(base_path, 'icon.ico')
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        else:
            print(f"Warning: Icon file not found at {icon_path}")  
        self._setup_ui()
        self.adjustSize()

    def _setup_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(10, 10, 10, 5)

        tabs = QTabWidget()
        main_layout.addWidget(tabs)
        controls_tab, log_tab = QWidget(), QWidget()
        tabs.addTab(controls_tab, "Playback Controls")
        tabs.addTab(log_tab, "Log Output")

        controls_layout = QVBoxLayout(controls_tab)
        controls_layout.addWidget(self._create_file_group())
        controls_layout.addWidget(self._create_playback_group())
        controls_layout.addWidget(self._create_humanization_group())
        controls_layout.addStretch()

        button_layout = QHBoxLayout()
        self.play_button = QPushButton("Play")
        self.stop_button = QPushButton("Stop")
        self.reset_button = QPushButton("Reset to Defaults")
        button_layout.addWidget(self.play_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addStretch()
        button_layout.addWidget(self.reset_button)
        main_layout.addLayout(button_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        main_layout.addWidget(self.progress_bar)

        self.play_button.clicked.connect(self.handle_play)
        self.stop_button.clicked.connect(self.handle_stop)
        self.reset_button.clicked.connect(self._reset_controls_to_default)

        log_layout = QVBoxLayout(log_tab)
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setFont(QFont("Courier", 9))
        log_layout.addWidget(self.log_output)

        self.stop_button.setEnabled(False)

    def _create_info_icon(self, tooltip_text: str) -> QLabel:
        label = QLabel("\u24D8")
        label.setStyleSheet("color: gray; font-weight: bold;")
        label.setToolTip(tooltip_text)
        return label

    def _create_slider_and_spinbox(self, min_val, max_val, default_val, text_suffix="", factor=10000.0, decimals=4):
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(int(min_val * factor), int(max_val * factor))
        
        spinbox = QDoubleSpinBox()
        spinbox.setDecimals(decimals)
        spinbox.setRange(0.0, 9999.9999) # Set a high upper bound for manual entry
        spinbox.setSingleStep(1.0 / factor)
        
        spinbox.setSuffix(text_suffix)
        slider.setValue(int(default_val * factor))
        spinbox.setValue(default_val)
        
        def slider_to_spinbox(value):
            spinbox.blockSignals(True)
            spinbox.setValue(value / factor)
            spinbox.blockSignals(False)

        def spinbox_to_slider(value):
            slider.blockSignals(True)
            slider.setValue(int(value * factor))
            slider.blockSignals(False)

        slider.valueChanged.connect(slider_to_spinbox)
        spinbox.valueChanged.connect(spinbox_to_slider)
        return slider, spinbox

    def _create_file_group(self):
        group = QGroupBox("MIDI File")
        layout = QVBoxLayout(group)
        self.file_path_label = QLabel("No file selected.")
        self.file_path_label.setStyleSheet("font-style: italic; color: grey;")
        browse_button = QPushButton("Browse for MIDI File")
        browse_button.clicked.connect(self.select_file)
        layout.addWidget(self.file_path_label)
        layout.addWidget(browse_button)
        return group

    def _create_playback_group(self):
        group = QGroupBox("Playback Settings")
        grid = QGridLayout(group)

        tempo_label = QLabel("Tempo:")
        tempo_info = self._create_info_icon("Adjusts the overall playback speed as a percentage of the original.\nExample: 120% plays the piece 20% faster, while 50% plays it at half speed.")
        tempo_slider, self.tempo_spinbox = self._create_slider_and_spinbox(0.1, 200.0, 100.0, "%", factor=100.0, decimals=1)
        grid.addWidget(tempo_label, 0, 0); grid.addWidget(tempo_info, 0, 1)
        grid.addWidget(tempo_slider, 0, 2); grid.addWidget(self.tempo_spinbox, 0, 3)

        pedal_label = QLabel("Pedal Style:")
        pedal_info = self._create_info_icon("Controls how the sustain pedal (spacebar) is automatically used.\n\n- Hybrid: Balances clarity and connection (recommended).\n- Legato: Connects notes smoothly, ideal for lyrical music.\n- Rhythmic: Emphasizes rhythmic patterns by pedaling on accented notes.\n- None: Disables automatic pedaling.")
        self.pedal_style_combo = QComboBox()
        self.pedal_style_combo.addItems(['hybrid', 'legato', 'rhythmic', 'none'])
        grid.addWidget(pedal_label, 1, 0); grid.addWidget(pedal_info, 1, 1)
        grid.addWidget(self.pedal_style_combo, 1, 2, 1, 2)

        self.piano_only_check = QCheckBox("Piano Only")
        self.piano_only_check.setToolTip("Use this for solo piano pieces. A sophisticated algorithm will attempt to assign notes\nto the left and right hands. If unchecked, a simple pitch split is used.")
        warning_label = QLabel("Limits to 10 notes played at once.")
        warning_label.setStyleSheet("color: #b8860b; font-style: italic;")
        warning_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        grid.addWidget(self.piano_only_check, 2, 0, 1, 2)
        grid.addWidget(warning_label, 2, 2, 1, 2)

        self.countdown_check = QCheckBox("Enable 3-second countdown")
        self.debug_check = QCheckBox("Enable debug output")
        grid.addWidget(self.countdown_check, 3, 0, 1, 4)
        grid.addWidget(self.debug_check, 4, 0, 1, 4)

        grid.setColumnStretch(2, 1)
        self._reset_playback_group_to_default()
        return group

    def _create_humanization_group(self):
        group = QGroupBox("Humanization Controls")
        grid = QGridLayout(group)
        self.humanization_sub_checkboxes.clear(); self.humanization_sliders.clear(); self.humanization_spinboxes.clear()

        self.select_all_humanization_check = QCheckBox("Select/Deselect All")
        self.select_all_humanization_check.setStyleSheet("font-weight: bold;")
        grid.addWidget(self.select_all_humanization_check, 0, 0, 1, 4)

        def add_humanization_row(row_idx, label, tip, min_val, max_val, def_val, suffix, factor=1.0, decimals=3):
            check = QCheckBox(label); info = self._create_info_icon(tip)
            slider, spinbox = self._create_slider_and_spinbox(min_val, max_val, def_val, suffix, factor=factor, decimals=decimals)
            grid.addWidget(check, row_idx, 0); grid.addWidget(info, row_idx, 1);
            grid.addWidget(slider, row_idx, 2); grid.addWidget(spinbox, row_idx, 3)
            check.toggled.connect(slider.setEnabled); check.toggled.connect(spinbox.setEnabled)
            self.humanization_sub_checkboxes.append(check); self.humanization_sliders.append(slider); self.humanization_spinboxes.append(spinbox)
            return check, spinbox

        add_humanization_row(1, "Timing Variance:", "Randomly alters note start times to simulate human timing imperfections.\nExample: A value of 0.010s (10ms) means notes can play up to 10ms earlier or later than written.", 0, 0.1, 0.01, " s", factor=10000.0)
        add_humanization_row(2, "Base Articulation:", "Sets the base length of every note as a percentage of its original duration.\nExample: 95% creates a slightly detached (staccato) feel, while 100% is fully connected (legato).", 50, 100, 95, "%", factor=100.0, decimals=1)
        add_humanization_row(3, "Hand Drift Decay:", "Simulates the natural timing drift between left and right hands.\nThis value controls how quickly the hands 're-sync' at musical phrase boundaries.\nHigher values mean faster re-synchronization.", 0, 100, 25, "%", factor=100.0, decimals=1)
        add_humanization_row(4, "Mistake Chance:", "Gives a percentage chance for a note to be 'mispressed' (a nearby note is played for a very\nshort duration) before the correct note sounds, simulating an accidental finger slip.", 0, 10, 0, "%", factor=100.0, decimals=1)
        add_humanization_row(5, "Tempo Sway:", "Simulates expressive tempo changes (rubato) over musical phrases.\nThis value is the maximum time shift in seconds. The actual sway is randomized per phrase\nand is amplified in slow sections and reduced in fast sections.", 0, 0.1, 0, " s", factor=10000.0)

        self.vary_velocity_check = QCheckBox("Vary note velocity")
        velocity_info = self._create_info_icon("Randomly adjusts the velocity (how 'hard' a key is pressed) of each note, making the performance sound more dynamic and less robotic.")
        grid.addWidget(self.vary_velocity_check, 6, 0); grid.addWidget(velocity_info, 6, 1)
        self.humanization_sub_checkboxes.append(self.vary_velocity_check)

        self.enable_chord_roll_check = QCheckBox("Enable chord rolling")
        roll_info = self._create_info_icon("Simulates the 'rolling' of chords, where the notes are played in very quick succession from bottom to top instead of at the exact same moment.")
        grid.addWidget(self.enable_chord_roll_check, 7, 0); grid.addWidget(roll_info, 7, 1)
        self.humanization_sub_checkboxes.append(self.enable_chord_roll_check)

        grid.setColumnStretch(2, 1)
        self.select_all_humanization_check.toggled.connect(self._toggle_all_humanization)
        for checkbox in self.humanization_sub_checkboxes: checkbox.toggled.connect(self._update_select_all_state)
        self._reset_humanization_group_to_default()
        return group

    def _reset_controls_to_default(self):
        self._reset_playback_group_to_default()
        self._reset_humanization_group_to_default()
        self.add_log_message("All settings have been reset to their default values.")

    def _reset_playback_group_to_default(self):
        self.tempo_spinbox.setValue(100); self.pedal_style_combo.setCurrentText('hybrid')
        self.piano_only_check.setChecked(False); self.countdown_check.setChecked(True)
        self.debug_check.setChecked(False)

    def _reset_humanization_group_to_default(self):
        self.humanization_spinboxes[0].setValue(0.010)
        self.humanization_spinboxes[1].setValue(95.0)
        self.humanization_spinboxes[2].setValue(25.0)
        self.humanization_spinboxes[3].setValue(0.0)
        self.humanization_spinboxes[4].setValue(0.0)
        self.select_all_humanization_check.setChecked(False)

    def _toggle_all_humanization(self, checked):
        for checkbox in self.humanization_sub_checkboxes: checkbox.setChecked(checked)

    def _update_select_all_state(self):
        is_all_checked = all(c.isChecked() for c in self.humanization_sub_checkboxes)
        self.select_all_humanization_check.blockSignals(True)
        self.select_all_humanization_check.setChecked(is_all_checked)
        self.select_all_humanization_check.blockSignals(False)

    def select_file(self):
        if self.player_thread and self.player_thread.isRunning(): return
        filepath, _ = QFileDialog.getOpenFileName(self, "Select MIDI File", "", "MIDI Files (*.mid *.midi)")
        if filepath:
            self.file_path_label.setText(filepath.split('/')[-1])
            self.file_path_label.setToolTip(filepath)
            self.file_path_label.setStyleSheet("")
            self.add_log_message(f"Selected file: {filepath}")

    def add_log_message(self, message): self.log_output.append(message)
    def update_progress(self, value): self.progress_bar.setValue(value)

    def set_controls_enabled(self, enabled):
        self.play_button.setEnabled(enabled); self.stop_button.setEnabled(not enabled)
        self.reset_button.setEnabled(enabled)
        for groupbox in self.findChildren(QGroupBox): groupbox.setEnabled(enabled)

    def gather_config(self) -> Optional[Dict]:
        filepath = self.file_path_label.toolTip()
        if not filepath:
            QMessageBox.warning(self, "No File", "Please select a MIDI file before playing."); return None
        return {
            'midi_file': filepath, 'tempo': self.tempo_spinbox.value(), 'countdown': self.countdown_check.isChecked(),
            'piano_only': self.piano_only_check.isChecked(), 'pedal_style': self.pedal_style_combo.currentText(),
            'debug_mode': self.debug_check.isChecked(),
            'vary_timing': self.humanization_sub_checkboxes[0].isChecked(), 'timing_variance': self.humanization_spinboxes[0].value(),
            'vary_articulation': self.humanization_sub_checkboxes[1].isChecked(), 'articulation': self.humanization_spinboxes[1].value() / 100.0,
            'enable_drift_correction': self.humanization_sub_checkboxes[2].isChecked(), 'drift_decay_factor': self.humanization_spinboxes[2].value() / 100.0,
            'enable_mistakes': self.humanization_sub_checkboxes[3].isChecked(), 'mistake_chance': self.humanization_spinboxes[3].value(),
            'enable_tempo_sway': self.humanization_sub_checkboxes[4].isChecked(), 'tempo_sway_intensity': self.humanization_spinboxes[4].value(),
            'vary_velocity': self.humanization_sub_checkboxes[5].isChecked(),
            'enable_chord_roll': self.humanization_sub_checkboxes[6].isChecked(),
        }

    def handle_play(self):
        if self.player_thread and self.player_thread.isRunning(): return
        config = self.gather_config()
        if not config: return
        self.set_controls_enabled(False)
        self.progress_bar.setValue(0)
        self.add_log_message("="*50 + f"\nStarting playback...")
        
        self.player_thread = QThread()
        self.player = Player(config)
        self.player.moveToThread(self.player_thread)
        self.player_thread.started.connect(self.player.play)
        self.player.playback_finished.connect(self.on_playback_finished)
        self.player.status_updated.connect(self.add_log_message)
        self.player.progress_updated.connect(self.update_progress)
        self.player_thread.start()

    def handle_stop(self):
        if self.player: self.player.stop()

    def on_playback_finished(self):
        self.add_log_message("Playback process finished.\n" + "="*50 + "\n")
        self.set_controls_enabled(True)
        if self.player_thread:
            self.player_thread.quit()
            self.player_thread.wait()
        self.player = None
        self.player_thread = None

    def closeEvent(self, event):
        if self.player and self.player_thread and self.player_thread.isRunning():
            self.add_log_message("Window closed during playback. Forcing stop...")
            self.player.stop()
            self.player_thread.wait(1000)
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
