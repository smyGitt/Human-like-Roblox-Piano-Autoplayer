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
from typing import List, Dict, Tuple, Optional, Set, Any
from collections import defaultdict
import os
from contextlib import ExitStack

from PyQt6.QtGui import QIcon, QGuiApplication

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
# Note: scikit-learn has been removed for a more lightweight package.
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
    pitch: Optional[int] = field(default=None, compare=False) # Add pitch for context

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
    def parse(filepath: str, tempo_scale: float = 1.0, debug_log: Optional[List[str]] = None) -> Tuple[List[Note], List[Tuple[float, int]]]:
        def _log(msg):
            if debug_log is not None: debug_log.append(f"[Parser] {msg}")
        
        try:
            mid = mido.MidiFile(filepath)
            _log(f"Successfully opened MIDI file: {filepath}")
        except Exception as e:
            raise IOError(f"Could not read or parse MIDI file: {e}")
            
        notes: List[Note] = []
        tempo_map_data: List[Tuple[float, int]] = []
        open_notes: Dict[int, List[Dict]] = defaultdict(list)
        absolute_time: float = 0.0
        tempo = 500000
        ticks_per_beat = mid.ticks_per_beat or 480
        _log(f"Ticks per beat: {ticks_per_beat}")
        
        tempo_map_data.append((0.0, tempo))
        note_id_counter = 0
        
        merged_track = mido.merge_tracks(mid.tracks)
        _log(f"Total MIDI messages to process: {len(merged_track)}")
        
        for i, msg in enumerate(merged_track):
            delta_time_sec = mido.tick2second(msg.time, ticks_per_beat, tempo)
            absolute_time += delta_time_sec
            _log(f"  Msg {i+1:<4} | Raw: {str(msg):<40} | Tick Δ: {msg.time:<5} | Time Δ: {delta_time_sec:7.4f}s | Absolute Time: {absolute_time:8.4f}s")
            
            if msg.type == 'set_tempo':
                tempo = msg.tempo
                tempo_map_data.append((absolute_time, tempo))
                _log(f"    -> TEMPO CHANGE to {mido.tempo2bpm(tempo):.2f} BPM")
            elif msg.type == 'note_on' and msg.velocity > 0:
                open_notes[msg.note].append({'start': absolute_time, 'vel': msg.velocity})
                _log(f"    -> NOTE ON  (Pitch: {msg.note}, Vel: {msg.velocity})")
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                _log(f"    -> NOTE OFF (Pitch: {msg.note})")
                if open_notes[msg.note]:
                    note_data = open_notes[msg.note].pop(0)
                    start = note_data['start']
                    duration = absolute_time - start
                    if duration > 0.01:
                        scaled_start = start / tempo_scale
                        scaled_duration = duration / tempo_scale
                        note = Note(id=note_id_counter, pitch=msg.note, velocity=note_data['vel'], start_time=scaled_start, duration=scaled_duration)
                        notes.append(note)
                        _log(f"      -> Created Note ID {note_id_counter}: Pitch {note.pitch}, Start {note.start_time:.4f}s, Dur {note.duration:.4f}s")
                        note_id_counter += 1
                    else:
                        _log(f"      -> Discarded note (duration {duration:.4f}s <= 0.01s)")
        
        notes.sort(key=lambda n: n.start_time)
        _log(f"Parsing complete. Found {len(notes)} valid notes.")
        return notes, tempo_map_data

class KeyMapper:
    # --- Layout Definitions ---
    LAYOUT_STANDARD = "1!2@34$5%6^78*9(0qQwWeErtTyYuiIoOpPasSdDfgGhHjJklLzZxcCvVbBnm"
    LAYOUT_88_LOWER = "1234567890qwert"
    LAYOUT_88_UPPER = "yuiopasdfghj"
    LAYOUT_88_FULL = LAYOUT_88_LOWER + LAYOUT_STANDARD + LAYOUT_88_UPPER

    SYMBOL_MAP = {'!': '1', '@': '2', '#': '3', '$': '4', '%': '5', '^': '6', '&': '7', '*': '8', '(': '9', ')': '0'}
    PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    BLACK_KEY_PITCH_CLASSES = {1, 3, 6, 8, 10}

    def __init__(self, use_88_key_layout: bool = False):
        self.use_88_key_layout = use_88_key_layout
        if self.use_88_key_layout:
            self.base_note = 21
            self.key_map = {self.base_note + i: key for i, key in enumerate(self.LAYOUT_88_FULL)}
            self.min_pitch, self.max_pitch = self.base_note, self.base_note + len(self.LAYOUT_88_FULL) - 1
            # Define bounds for modifier logic
            self.lower_ctrl_bound = self.base_note + len(self.LAYOUT_88_LOWER)
            self.upper_ctrl_bound = self.base_note + len(self.LAYOUT_88_LOWER) + len(self.LAYOUT_STANDARD)
        else:
            c4_index = self.LAYOUT_STANDARD.find('t')
            self.base_note = 60 - c4_index
            self.key_map = {self.base_note + i: key for i, key in enumerate(self.LAYOUT_STANDARD)}
            self.min_pitch, self.max_pitch = self.base_note, self.base_note + len(self.LAYOUT_STANDARD) - 1
            self.lower_ctrl_bound = -1 # Not used
            self.upper_ctrl_bound = -1 # Not used

    def get_key_for_pitch(self, pitch: int) -> Optional[str]:
        if self.min_pitch <= pitch <= self.max_pitch:
            return self.key_map.get(pitch)
        
        if self.use_88_key_layout:
            return None

        transposed_pitch = pitch
        if transposed_pitch < self.min_pitch:
            while transposed_pitch < self.min_pitch: transposed_pitch += 12
        elif transposed_pitch > self.max_pitch:
            while transposed_pitch > self.max_pitch: transposed_pitch -= 12
        return self.key_map.get(transposed_pitch)

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
    def __init__(self, config: Dict, debug_log: Optional[List[str]] = None):
        self.config = config
        self.debug_log = debug_log
        self.left_hand_drift = 0.0
        self.right_hand_drift = 0.0

    def _log(self, msg):
        if self.debug_log is not None: self.debug_log.append(f"[Humanizer] {msg}")

    def apply_to_hand(self, notes: List[Note], hand: str, resync_points: Set[float]):
        self._log(f"Applying humanization to {hand} hand ({len(notes)} notes)...")
        if not any([self.config.get('vary_timing'), self.config.get('vary_articulation'), self.config.get('vary_velocity'), self.config.get('enable_drift_correction'), self.config.get('enable_chord_roll')]):
            self._log("  -> No humanization options enabled for this hand.")
            return
            
        time_groups = get_time_groups(notes)
        for group in time_groups:
            is_resync_point = round(group[0].start_time, 2) in resync_points
            if self.config.get('enable_drift_correction') and is_resync_point:
                self._log(f"  Resync point at {group[0].start_time:.4f}s. Decaying drift.")
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
                for i, note in enumerate(group):
                    roll_offset = i * 0.006
                    note.start_time += roll_offset
                    self._log(f"    Note {note.id} rolled by +{roll_offset:.4f}s")

            for note in group:
                log_prefix = f"    Note {note.id:<4} ({KeyMapper.pitch_to_name(note.pitch):<4}):"
                original_start, original_dur = note.start_time, note.duration

                current_drift = self.left_hand_drift if hand == 'left' else self.right_hand_drift
                note.start_time += group_timing_offset
                if self.config.get('enable_drift_correction'):
                    note.start_time += current_drift
                
                note.duration *= group_articulation
                
                if self.config.get('vary_velocity'):
                    vel_multiplier = (1 + (random.random() - 0.5) * 0.2)
                    note.velocity = max(1, int(note.velocity * vel_multiplier))
                
                self._log(f"{log_prefix} Time {original_start:8.4f} -> {note.start_time:8.4f} | Dur {original_dur:7.4f} -> {note.duration:7.4f}")

            if self.config.get('enable_drift_correction'):
                if hand == 'left': self.left_hand_drift += group_timing_offset
                else: self.right_hand_drift += group_timing_offset

    def apply_tempo_rubato(self, all_notes: List[Note], sections: List[MusicalSection]):
        if not self.config.get('enable_tempo_sway'):
            return
        self._log("Applying tempo rubato (sway)...")
        base_intensity = self.config.get('tempo_sway_intensity', 0.0)
        note_map = {note.id: note for note in all_notes}

        for section in sections:
            if section.pace_label == 'fast': pace_multiplier = 0.25
            elif section.pace_label == 'slow': pace_multiplier = 1.5
            else: pace_multiplier = 1.0

            for phrase in section.rhythmic_phrases:
                phrase_duration = phrase.end_time - phrase.start_time
                if phrase_duration < 1.0: continue
                phrase_intensity = random.uniform(0.5, 1.0) * base_intensity * pace_multiplier
                if phrase_intensity == 0: continue
                phase_shift = random.uniform(-np.pi / 4, np.pi / 4)
                self._log(f"  Swaying phrase at {phrase.start_time:.3f}s with intensity {phrase_intensity:.4f}")

                for note_in_phrase in phrase.notes:
                    if note_in_phrase.id in note_map:
                        note_to_modify = note_map[note_in_phrase.id]
                        relative_pos = (note_to_modify.start_time - phrase.start_time) / phrase_duration
                        time_shift = np.sin(relative_pos * np.pi + phase_shift) * phrase_intensity
                        note_to_modify.start_time -= time_shift


class FingeringEngine:
    TRAVEL_WEIGHT, RECENCY_WEIGHT, STRETCH_WEIGHT = 1.0, 150.0, 0.5
    CROSSOVER_PENALTY, THUMB_ON_BLACK_KEY_PENALTY = 50.0, 20.0
    MAX_HAND_SPAN = 14
    def __init__(self, debug_log: Optional[List[str]] = None):
        self.fingers = [Finger(id=i, hand='left') for i in range(5)] + [Finger(id=i, hand='right') for i in range(5, 10)]
        self.debug_log = debug_log

    def _log(self, msg):
        if self.debug_log is not None: self.debug_log.append(f"[Fingering] {msg}")

    def assign_hands(self, notes: List[Note]):
        self._log("Starting advanced hand assignment...")
        time_groups = get_time_groups(notes)
        for i, group in enumerate(time_groups):
            self._log(f"  Processing time group {i+1}/{len(time_groups)} at {group[0].start_time:.3f}s ({len(group)} notes)")
            if len(group) == 1: self._assign_single_note(group[0])
            else: self._assign_chord(group)
        self._log("Hand assignment complete.")

    def _update_finger_state(self, finger: Finger, note: Note):
        finger.current_pitch = note.pitch; finger.last_press_time = note.start_time
    
    def _calculate_cost(self, finger: Finger, note: Note) -> float:
        if finger.current_pitch is None: return 0
        travel_cost = abs(finger.current_pitch - note.pitch) * self.TRAVEL_WEIGHT
        recency_cost = 0
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
        
        total_cost = travel_cost + recency_cost + thumb_cost + stretch_cost + crossover_cost
        self._log(f"      Cost for Finger {finger.id} ({finger.hand}): Total={total_cost:.1f} (Travel={travel_cost:.1f}, Recency={recency_cost:.1f}, Thumb={thumb_cost:.1f}, Stretch={stretch_cost:.1f}, Crossover={crossover_cost:.1f})")
        return total_cost

    def _assign_single_note(self, note: Note):
        self._log(f"    Assigning single note {note.id} ({KeyMapper.pitch_to_name(note.pitch)})...")
        costs = [(self._calculate_cost(f, note), f) for f in self.fingers]
        _, best_finger = min(costs, key=lambda x: x[0])
        note.hand = best_finger.hand
        self._log(f"    -> Assigned to {best_finger.hand} hand (Finger {best_finger.id}).")
        self._update_finger_state(best_finger, note)

    def _assign_chord(self, chord_notes: List[Note]):
        self._log(f"    Assigning chord of {len(chord_notes)} notes...")
        chord_notes.sort(key=lambda n: n.pitch)
        span = chord_notes[-1].pitch - chord_notes[0].pitch
        if span > self.MAX_HAND_SPAN:
            self._log(f"      Chord span ({span}) > max hand span ({self.MAX_HAND_SPAN}). Splitting between hands.")
            best_gap, split_index = -1, -1
            for i in range(len(chord_notes) - 1):
                gap = chord_notes[i+1].pitch - chord_notes[i].pitch
                if gap > best_gap: best_gap, split_index = gap, i + 1
            for i, note in enumerate(chord_notes):
                note.hand = 'left' if i < split_index else 'right'
                self._log(f"      Note {note.id} ({KeyMapper.pitch_to_name(note.pitch)}) -> {note.hand} hand.")
        else:
            left_fingers = [f for f in self.fingers if f.hand == 'left' and f.current_pitch is not None]
            right_fingers = [f for f in self.fingers if f.hand == 'right' and f.current_pitch is not None]
            left_center = np.mean([f.current_pitch for f in left_fingers]) if left_fingers else 48
            right_center = np.mean([f.current_pitch for f in right_fingers]) if right_fingers else 72
            chord_center = np.mean([n.pitch for n in chord_notes])
            chosen_hand = 'left' if abs(chord_center - left_center) <= abs(chord_center - right_center) else 'right'
            self._log(f"      Chord span ({span}) within range. Assigning to one hand. Chord Center: {chord_center:.1f}, Left Hand Center: {left_center:.1f}, Right Hand Center: {right_center:.1f} -> Chose {chosen_hand} hand.")
            for note in chord_notes: note.hand = chosen_hand
        
        for note in chord_notes:
            hand_fingers = [f for f in self.fingers if f.hand == note.hand]
            for f in hand_fingers: f.last_press_time = note.start_time
            closest_finger = min(hand_fingers, key=lambda f: abs(f.current_pitch - note.pitch) if f.current_pitch else float('inf'))
            closest_finger.current_pitch = note.pitch

class PedalGenerator:
    @staticmethod
    def generate_events(config: Dict, final_notes: List[Note], sections: List[MusicalSection], debug_log: Optional[List[str]] = None) -> List[KeyEvent]:
        def _log(msg):
            if debug_log is not None: debug_log.append(f"[Pedal] {msg}")

        style = config.get('pedal_style')
        if style == 'none' or not final_notes: 
            _log("Pedal style is 'none' or no notes provided. Skipping pedal generation.")
            return []
        
        _log(f"Generating pedal events with style: '{style}'")
        events = []
        for sec in sections:
            if sec.is_bridge:
                sec_final_notes = [n for n in final_notes if sec.start_time <= n.start_time < sec.end_time]
                PedalGenerator._apply_bridge_pedal(events, get_time_groups(sec_final_notes)); continue
            for phrase in sec.rhythmic_phrases:
                phrase_final_notes = [n for n in final_notes if phrase.start_time <= n.start_time < phrase.end_time]
                phrase_time_groups = get_time_groups(phrase_final_notes);
                if not phrase_time_groups: continue
                
                _log(f"  Processing phrase at {phrase.start_time:.3f}s (Artic: {phrase.articulation_label}, Pattern: {phrase.pattern_label})")
                
                if phrase.pattern_label == 'arpeggio': 
                    _log("    -> Applying bridge pedal for arpeggio.")
                    PedalGenerator._apply_bridge_pedal(events, phrase_time_groups); continue
                elif phrase.pattern_label in ['scale', 'ornament']: 
                    _log("    -> Skipping pedal for scale/ornament.")
                    continue
                
                if style == 'hybrid':
                    if phrase.articulation_label in ['staccato', 'staccatissimo', 'tenuto']: 
                        _log("    -> Applying accent pedal for rhythmic articulation.")
                        PedalGenerator._apply_accent_pedal(events, phrase_time_groups)
                    else: 
                        _log("    -> Applying clarity legato pedal.")
                        PedalGenerator._apply_clarity_legato(events, phrase_time_groups)
                elif style == 'legato':
                    if phrase.articulation_label in ['legato', 'tenuto', 'uniform']: 
                        _log("    -> Applying clarity legato pedal.")
                        PedalGenerator._apply_clarity_legato(events, phrase_time_groups)
                elif style == 'rhythmic':
                    if phrase.articulation_label in ['staccato', 'staccatissimo', 'tenuto']: 
                        _log("    -> Applying accent pedal for rhythmic articulation.")
                        PedalGenerator._apply_accent_pedal(events, phrase_time_groups)
        
        _log(f"Generated {len(events)} pedal events.")
        return events

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

class SectionAnalyzer:
    ARTICULATION_LABELS = { 1: ['uniform'], 2: ['legato', 'staccato'], 3: ['legato', 'tenuto', 'staccato'], 4: ['legato', 'tenuto', 'staccato', 'staccatissimo'] }
    def __init__(self, notes: List[Note], tempo_map: TempoMap, debug_log: Optional[List[str]] = None):
        self.notes, self.tempo_map = notes, tempo_map
        self.time_groups = get_time_groups(notes)
        self.debug_log = debug_log

    def _log(self, msg):
        if self.debug_log is not None: self.debug_log.append(f"[Analyzer] {msg}")

    def _lightweight_kmeans(self, data: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """A lightweight, pure Python/Numpy implementation of KMeans."""
        # 1. Initialize centroids randomly from the data points
        indices = np.random.choice(data.shape[0], k, replace=False)
        centroids = data[indices]
        
        for _ in range(100): # Max 100 iterations to prevent infinite loops
            # 2. Assign clusters
            distances = np.sqrt(((data - centroids[:, np.newaxis])**2).sum(axis=2))
            labels = np.argmin(distances, axis=0)
            
            # 3. Update centroids
            new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])
            
            if np.all(centroids == new_centroids):
                break
            centroids = new_centroids
            
        return labels, centroids

    def analyze(self) -> List[MusicalSection]:
        self._log("Starting musical analysis...")
        if not self.time_groups: 
            self._log("No time groups found, cannot analyze.")
            return []
        
        phrase_boundaries = self._plan_phrases_with_bridges();
        self._log(f"Found {len(phrase_boundaries)} initial phrase boundaries.")
        if not phrase_boundaries: return []
        
        initial_phrases = self._create_sections_from_boundaries(phrase_boundaries)
        merged_sections = self._merge_similar_sections(initial_phrases)
        self._log(f"Merged similar sections, resulting in {len(merged_sections)} major sections.")
        
        classified_sections = self._classify_sections_by_pace(merged_sections)
        for section in classified_sections:
            if section.is_bridge: 
                self._finalize_rhythmic_phrase(section, get_time_groups(section.notes), 'bridge_held', 'standard'); continue
            self._subdivide_section_by_articulation(section)
        
        self._log("Analysis complete.")
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
        if not sections: return []
        self._log("Classifying sections by pace...")
        if len(sections) < 3:
            for section in sections: section.pace_label = 'normal'
            self._log("  -> Not enough sections for statistical classification, all labeled 'normal'.")
            return sections
        densities = [s.normalized_density for s in sections]; slow_q, fast_q = np.percentile(densities, [33, 67])
        self._log(f"  -> Pace quantiles (33%, 67%): {slow_q:.2f}, {fast_q:.2f}")
        for section in sections:
            if section.normalized_density <= slow_q: section.pace_label = 'slow'
            elif section.normalized_density >= fast_q: section.pace_label = 'fast'
            else: section.pace_label = 'normal'
            self._log(f"  Section at {section.start_time:.3f}s labeled as '{section.pace_label}' (density: {section.normalized_density:.2f})")
        return sections
    def _subdivide_section_by_articulation(self, section: MusicalSection):
        time_groups = get_time_groups(section.notes)
        if len(time_groups) < 3: 
            self._finalize_rhythmic_phrase(section, time_groups, 'uniform', self._detect_pattern(section.notes)); return
        inter_chord_gaps = []
        for i in range(len(time_groups) - 1):
            gap_start = max(n.start_time + n.duration for n in time_groups[i]); gap_end = time_groups[i+1][0].start_time
            tempo = self.tempo_map.get_tempo_at(gap_start); beat_duration = tempo / 1_000_000.0
            normalized_gap = (gap_end - gap_start) / beat_duration if beat_duration > 0 else 0
            inter_chord_gaps.append(max(0, normalized_gap))
        
        gap_data = np.array(inter_chord_gaps).reshape(-1, 1)
        unique_gaps = np.unique(gap_data)
        if len(unique_gaps) < 2:
            self._finalize_rhythmic_phrase(section, time_groups, 'uniform', self._detect_pattern(section.notes)); return

        # Determine optimal k for KMeans
        max_k = min(len(unique_gaps), 4)
        if max_k < 2:
            self._finalize_rhythmic_phrase(section, time_groups, 'uniform', self._detect_pattern(section.notes)); return

        inertias = []
        for k_test in range(1, max_k + 1):
            _, centroids = self._lightweight_kmeans(gap_data, k_test)
            distances = np.min(np.sqrt(((gap_data - centroids[:, np.newaxis])**2).sum(axis=2)), axis=0)
            inertias.append(np.sum(distances**2))

        optimal_k = 1
        if len(inertias) > 1:
            diffs = np.diff(inertias)
            if len(diffs) > 1: optimal_k = np.argmax(np.diff(diffs)) + 2
            else: optimal_k = 2
        
        # Run final KMeans with optimal k
        labels, centroids = self._lightweight_kmeans(gap_data, optimal_k)
        
        sorted_indices = np.argsort(centroids.flatten())
        labels_for_k = self.ARTICULATION_LABELS.get(optimal_k, self.ARTICULATION_LABELS[4])
        label_map = {idx: label for idx, label in zip(sorted_indices, labels_for_k)}
        gap_labels = [label_map[label] for label in labels]
        
        current_phrase_groups, current_label = [time_groups[0]], gap_labels[0]
        for i, label in enumerate(gap_labels):
            if label == current_label: current_phrase_groups.append(time_groups[i+1])
            else:
                phrase_notes = [n for g in current_phrase_groups for n in g]
                self._finalize_rhythmic_phrase(section, current_phrase_groups, current_label, self._detect_pattern(phrase_notes))
                current_phrase_groups, current_label = [time_groups[i+1]], label
        phrase_notes = [n for g in current_phrase_groups for n in g]
        self._finalize_rhythmic_phrase(section, current_phrase_groups, current_label, self._detect_pattern(phrase_notes))

    def _finalize_rhythmic_phrase(self, section: MusicalSection, phrase_groups: List[List[Note]], articulation: str, pattern: str):
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
        self.mapper = KeyMapper(use_88_key_layout=self.config.get('use_88_key_layout', False))
        self.event_queue: List[KeyEvent] = []
        self.stop_event = threading.Event()
        self.key_states: Dict[str, KeyState] = {}
        self.pedal_is_down = False
        self.tempo_map = None
        self.debug_log: Optional[List[str]] = [] if self.config.get('debug_mode') else None
    
    def _log_debug(self, msg: str):
        if self.debug_log is not None:
            self.debug_log.append(msg)

    def play(self):
        try:
            self._log_debug("--- STARTING PLAYBACK GENERATION ---")
            self._log_debug("\n[1. Configuration]")
            for key, val in self.config.items():
                self._log_debug(f"  - {key}: {val}")
            
            self._log_debug("\n[2. Parsing & Analysis]")
            original_notes, sections = self._initialize_and_analyze()
            if not original_notes:
                self.status_updated.emit("Error: No playable notes found in the file.")
                self.playback_finished.emit()
                return

            self._log_debug("\n[3. Humanization]")
            humanized_notes = copy.deepcopy(original_notes)
            self.humanizer = Humanizer(self.config, self.debug_log)
            left_hand_notes = [n for n in humanized_notes if n.hand == 'left']
            right_hand_notes = [n for n in humanized_notes if n.hand == 'right']
            resync_points = {round(n.start_time, 2) for n in left_hand_notes}.intersection({round(n.start_time, 2) for n in right_hand_notes})

            self.humanizer.apply_to_hand(left_hand_notes, 'left', resync_points)
            self.humanizer.apply_to_hand(right_hand_notes, 'right', resync_points)
            
            all_notes = sorted(left_hand_notes + right_hand_notes, key=lambda n: n.start_time)
            
            self.humanizer.apply_tempo_rubato(all_notes, sections)
            
            self._log_debug("\n[4. Event Scheduling]")
            self._schedule_events(all_notes, sections)
            
            self._log_debug("\n[5. Final Event Queue]")
            temp_queue = sorted(list(self.event_queue))
            for event in temp_queue:
                self._log_debug(f"  - Time: {event.time:8.4f}s | Prio: {event.priority} | Action: {event.action:<7} | Key: '{event.key_char}' | Pitch: {event.pitch}")

            self._log_debug("\n--- PLAYBACK GENERATION COMPLETE ---")
            if self.debug_log is not None:
                self.status_updated.emit("\n".join(self.debug_log))

            if self.config.get('countdown'): self._run_countdown()
            if self.stop_event.is_set():
                self.playback_finished.emit()
                return

            self.status_updated.emit("Playback starting...")
            self._run_scheduler()

        except (IOError, ValueError) as e:
            self.status_updated.emit(f"Error: {e}")
        except Exception as e:
            import traceback
            self.status_updated.emit(f"An unexpected error occurred: {e}\n{traceback.format_exc()}")
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
        original_notes, tempo_data = self.parser.parse(self.config.get('midi_file'), tempo_scale, self.debug_log)
        if not original_notes: return None, None
        
        self._log_debug("\n[2a. Hand Assignment]")
        if self.config.get('simulate_hands'):
            self.status_updated.emit("Using advanced hand simulation model.")
            self._log_debug("Method: Advanced Fingering Engine")
            engine = FingeringEngine(self.debug_log)
            engine.assign_hands(original_notes)
        else:
            self.status_updated.emit("Using simple pitch-split for hand assignment.")
            self._log_debug("Method: Simple Pitch Split (at pitch 60)")
            self._separate_hands_by_pitch(original_notes)

        self.tempo_map = TempoMap(tempo_data)
        duration = max(n.start_time + n.duration for n in original_notes) if original_notes else 0
        self.status_updated.emit(f"Loaded {len(original_notes)} notes. Estimated duration: {duration:.2f}s")

        self.status_updated.emit("Analyzing musical structure...")
        self._log_debug("\n[2b. Musical Structure Analysis]")
        analyzer = SectionAnalyzer(original_notes, self.tempo_map, self.debug_log)
        sections = analyzer.analyze()
        self.status_updated.emit(f"Analysis complete. Found {len(sections)} major sections.")

        return original_notes, sections

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
        
        self._log_debug("Scheduling note and mistake events...")
        for note in notes_to_play:
            note_str = f"Note ID {note.id:<4} ({self.mapper.pitch_to_name(note.pitch):<4}, pitch {note.pitch:<3}, vel {note.velocity:<3}) at {note.start_time:.4f}s"
            
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
                        self._log_debug(f"  {note_str} -> MISTAKE generated (slip to {self.mapper.pitch_to_name(mistake_pitch)})")
                        if note.duration > mistake_duration: # Slip-and-correct mistake
                            heapq.heappush(self.event_queue, KeyEvent(note.start_time, 2, 'press', mistake_key, pitch=mistake_pitch))
                            heapq.heappush(self.event_queue, KeyEvent(note.start_time + mistake_duration, 4, 'release', mistake_key, pitch=mistake_pitch))
                            correct_start = note.start_time + mistake_duration
                            correct_duration = note.duration - mistake_duration
                            heapq.heappush(self.event_queue, KeyEvent(correct_start, 2, 'press', correct_key, pitch=note.pitch))
                            heapq.heappush(self.event_queue, KeyEvent(correct_start + correct_duration, 4, 'release', correct_key, pitch=note.pitch))
                        else: # "Fatal" mistake for short notes
                            heapq.heappush(self.event_queue, KeyEvent(note.start_time, 2, 'press', mistake_key, pitch=mistake_pitch))
                            heapq.heappush(self.event_queue, KeyEvent(note.start_time + note.duration, 4, 'release', mistake_key, pitch=mistake_pitch))
                        
                        if correct_key not in self.key_states: self.key_states[correct_key] = KeyState(correct_key)
                        if mistake_key not in self.key_states: self.key_states[mistake_key] = KeyState(mistake_key)
                        mistake_scheduled = True

            if not mistake_scheduled:
                key_char = self.mapper.get_key_for_pitch(note.pitch)
                if key_char:
                    self._log_debug(f"  {note_str} -> Mapped to key '{key_char}'")
                    heapq.heappush(self.event_queue, KeyEvent(note.start_time, 2, 'press', key_char, pitch=note.pitch))
                    heapq.heappush(self.event_queue, KeyEvent(note.start_time + note.duration, 4, 'release', key_char, pitch=note.pitch))
                    if key_char not in self.key_states:
                        self.key_states[key_char] = KeyState(key_char)
                else:
                    self._log_debug(f"  {note_str} -> SKIPPED (no valid key in current layout)")
            
            played_pitches_in_section.add(note.pitch)
        
        self._log_debug("\n[4a. Pedal Generation]")
        pedal_events = PedalGenerator.generate_events(self.config, notes_to_play, sections, self.debug_log)
        for event in pedal_events:
            heapq.heappush(self.event_queue, event)
        self._log_debug("All events scheduled.")
            
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
        self._log_debug("\n[6. Real-time Playback Log]")
        start_time = time.perf_counter()
        total_duration = max(e.time for e in self.event_queue) if self.event_queue else 0

        while not self.stop_event.is_set() and self.event_queue:
            playback_time = time.perf_counter() - start_time
            
            if self.event_queue[0].time <= playback_time:
                current_time = self.event_queue[0].time
                events_to_process = []
                
                while self.event_queue and self.event_queue[0].time == current_time:
                    events_to_process.append(heapq.heappop(self.event_queue))
                
                events_to_process.sort(key=lambda e: e.priority)
                self._execute_chord_event(events_to_process, playback_time)
            else:
                time.sleep(0.001)
            
            if total_duration > 0:
                progress = int((playback_time / total_duration) * 100)
                self.progress_updated.emit(progress)

    def _get_press_info_from_event(self, event: KeyEvent) -> Tuple[List[Key], str]:
        """Determines modifiers and base key from a KeyEvent, using its pitch."""
        modifiers = []
        key_char = event.key_char
        pitch = event.pitch
        base_key = key_char.lower()

        if self.config.get('use_88_key_layout') and pitch is not None:
            # Check for Ctrl based on PITCH ranges
            if pitch < self.mapper.lower_ctrl_bound or pitch >= self.mapper.upper_ctrl_bound:
                modifiers.append(Key.ctrl)
        
        # Check for Shift based on CHARACTER (symbol or uppercase)
        if key_char in self.mapper.SYMBOL_MAP:
            modifiers.append(Key.shift)
            base_key = self.mapper.SYMBOL_MAP[key_char]
        elif key_char.isupper():
            modifiers.append(Key.shift)
            
        return modifiers, base_key

    def _execute_chord_event(self, events: List[KeyEvent], playback_time: float):
        if self.stop_event.is_set(): return

        press_events = [e for e in events if e.action == 'press']
        release_events = [e for e in events if e.action == 'release']
        pedal_events = [e for e in events if e.action == 'pedal']
        
        self._log_debug(f"T={playback_time:8.4f}s | Processing event group (P:{len(press_events)}, R:{len(release_events)}, Pdl:{len(pedal_events)})")

        # 1. Handle Pedal Events first
        for event in pedal_events:
            self._log_debug(f"  -> Executing: {event.action.upper()} '{event.key_char}'")
            self._handle_pedal_event(event)

        # 2. Handle Releases
        for event in release_events:
            self._log_debug(f"  -> Executing: {event.action.upper()} '{event.key_char}'")
            key_char = event.key_char
            state = self.key_states.get(key_char)
            if not state: continue
            
            # Release logic doesn't need complex modifier checks, just the base key
            base_key = key_char.lower()
            if key_char in self.mapper.SYMBOL_MAP:
                base_key = self.mapper.SYMBOL_MAP[key_char]

            was_physically_down = state.is_physically_down
            state.release(self.pedal_is_down)
            
            if was_physically_down and not state.is_physically_down:
                self._log_debug(f"    - Releasing physical key '{base_key}'.")
                try: self.keyboard.release(base_key)
                except Exception: pass

        # 3. Handle Presses individually to manage modifiers correctly
        for event in press_events:
            self._log_debug(f"  -> Executing: {event.action.upper()} '{event.key_char}' (Pitch: {event.pitch})")
            state = self.key_states.get(event.key_char)
            if not state or event.pitch is None: continue
            
            modifiers, base_key = self._get_press_info_from_event(event)
            
            was_physically_down = state.is_physically_down
            is_sustained_only = state.is_sustained and not state.is_active
            state.press()
            
            try:
                # Use a context manager for each key's specific modifiers
                with self.keyboard.pressed(*modifiers):
                    if is_sustained_only:
                        self._log_debug(f"    - Re-pressing sustained key. Releasing '{base_key}', then pressing with {modifiers}.")
                        self.keyboard.release(base_key)
                        time.sleep(0.001) # Small delay to ensure physical release
                        self.keyboard.press(base_key)
                    elif not was_physically_down:
                        self._log_debug(f"    - Pressing physical key '{base_key}' with modifiers {modifiers}.")
                        self.keyboard.press(base_key)
            except Exception as e:
                self._log_debug(f"    - ERROR during key press: {e}")

    def _handle_pedal_event(self, event: KeyEvent):
        if self.stop_event.is_set(): return
        
        if event.key_char == 'down' and not self.pedal_is_down:
            self.pedal_is_down = True
            self._log_debug(f"    - Pedal State: DOWN. Pressing physical spacebar.")
            try: self.keyboard.press(Key.space)
            except Exception: pass
        elif event.key_char == 'up' and self.pedal_is_down:
            self.pedal_is_down = False
            self._log_debug(f"    - Pedal State: UP. Releasing physical spacebar.")
            try: self.keyboard.release(Key.space)
            except Exception: pass
            
            self._log_debug(f"    - Lifting sustain from all keys.")
            for key_char, state in self.key_states.items():
                was_physically_down = state.is_physically_down
                state.lift_sustain()
                if was_physically_down and not state.is_physically_down:
                    try:
                        # Use the simple character-based logic for releases
                        base_key = key_char.lower()
                        if key_char in self.mapper.SYMBOL_MAP:
                            base_key = self.mapper.SYMBOL_MAP[key_char]
                        self._log_debug(f"      -> Releasing sustained key '{key_char}' (physical key '{base_key}')")
                        self.keyboard.release(base_key)
                    except Exception: pass

    def shutdown(self):
        self.status_updated.emit("Releasing all keys...")
        self._log_debug("\n[7. Shutdown]")
        for key_char, state in list(self.key_states.items()):
            if state.is_physically_down:
                try:
                    base_key = key_char.lower()
                    if key_char in self.mapper.SYMBOL_MAP:
                        base_key = self.mapper.SYMBOL_MAP[key_char]
                    self._log_debug(f"  Releasing active key '{key_char}'")
                    self.keyboard.release(base_key)
                except Exception: pass
        self.key_states.clear()
        if self.pedal_is_down:
            self._log_debug("  Releasing pedal (spacebar).")
            try: self.keyboard.release(Key.space)
            except Exception: pass
        for key in [Key.shift, Key.ctrl, Key.alt]:
            self._log_debug(f"  Releasing modifier key {key}.")
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

        # --- Log Tab Setup ---
        log_layout = QVBoxLayout(log_tab)
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setFont(QFont("Courier", 9))
        log_layout.addWidget(self.log_output)
        
        log_button_layout = QHBoxLayout()
        clear_log_button = QPushButton("Clear Log")
        clear_log_button.clicked.connect(self.clear_log)
        log_button_layout.addWidget(clear_log_button)
        log_button_layout.addStretch()
        copy_log_button = QPushButton("Copy to Clipboard")
        copy_log_button.clicked.connect(self.copy_log_to_clipboard)
        log_button_layout.addWidget(copy_log_button)
        log_layout.addLayout(log_button_layout)


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

        # Row 0: Tempo
        tempo_label = QLabel("Tempo:")
        tempo_info = self._create_info_icon("Adjusts the overall playback speed as a percentage of the original.\nExample: 120% plays the piece 20% faster, while 50% plays it at half speed.")
        tempo_slider, self.tempo_spinbox = self._create_slider_and_spinbox(10.0, 200.0, 100.0, "%", factor=10.0, decimals=1)
        grid.addWidget(tempo_label, 0, 0); grid.addWidget(tempo_info, 0, 1)
        grid.addWidget(tempo_slider, 0, 2); grid.addWidget(self.tempo_spinbox, 0, 3)

        # Row 1: Pedal Style
        pedal_label = QLabel("Pedal Style:")
        pedal_info = self._create_info_icon("Controls how the sustain pedal (spacebar) is automatically used.\n\n- Hybrid: Balances clarity and connection (recommended).\n- Legato: Connects notes smoothly, ideal for lyrical music.\n- Rhythmic: Emphasizes rhythmic patterns by pedaling on accented notes.\n- None: Disables automatic pedaling.")
        self.pedal_style_combo = QComboBox()
        self.pedal_style_combo.addItems(['hybrid', 'legato', 'rhythmic', 'none'])
        grid.addWidget(pedal_label, 1, 0); grid.addWidget(pedal_info, 1, 1)
        grid.addWidget(self.pedal_style_combo, 1, 2, 1, 2)

        # Row 2: 88-Key Layout
        self.use_88_key_check = QCheckBox("Use 88-Key Extended Layout")
        self.use_88_key_check.setToolTip("Expands the keyboard mapping to cover a full 88-key piano range.\nThis disables automatic octave transposition and uses 'Ctrl + key' for the lowest and highest octaves.")
        grid.addWidget(self.use_88_key_check, 2, 0, 1, 4)

        # Row 3 & 4: Other Checkboxes
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
        
        # --- Simulate Hands Checkbox ---
        self.simulate_hands_check = QCheckBox("Simulate hands")
        self.simulate_hands_check.setToolTip("Use this for solo piano pieces. A sophisticated algorithm will attempt to assign notes\nto the left and right hands, simulating a human player. If unchecked, a simple pitch split is used.")
        info_hands = self._create_info_icon("Limits to 10 notes at once (5 per hand).")
        grid.addWidget(self.simulate_hands_check, 1, 0)
        grid.addWidget(info_hands, 1, 1)
        self.humanization_sub_checkboxes.append(self.simulate_hands_check)


        def add_humanization_row(row_idx, label, tip, min_val, max_val, def_val, suffix, factor=1.0, decimals=3):
            check = QCheckBox(label); info = self._create_info_icon(tip)
            slider, spinbox = self._create_slider_and_spinbox(min_val, max_val, def_val, suffix, factor=factor, decimals=decimals)
            grid.addWidget(check, row_idx, 0); grid.addWidget(info, row_idx, 1);
            grid.addWidget(slider, row_idx, 2); grid.addWidget(spinbox, row_idx, 3)
            check.toggled.connect(slider.setEnabled); check.toggled.connect(spinbox.setEnabled)
            self.humanization_sub_checkboxes.append(check); self.humanization_sliders.append(slider); self.humanization_spinboxes.append(spinbox)
            return check, spinbox

        add_humanization_row(2, "Timing Variance:", "Randomly alters note start times to simulate human timing imperfections.\nExample: A value of 0.010s (10ms) means notes can play up to 10ms earlier or later than written.", 0, 0.1, 0.01, " s", factor=10000.0)
        add_humanization_row(3, "Base Articulation:", "Sets the base length of every note as a percentage of its original duration.\nExample: 95% creates a slightly detached (staccato) feel, while 100% is fully connected (legato).", 50, 100, 95, "%", factor=100.0, decimals=1)
        add_humanization_row(4, "Hand Drift Decay:", "Simulates the natural timing drift between left and right hands.\nThis value controls how quickly the hands 're-sync' at musical phrase boundaries.\nHigher values mean faster re-synchronization.", 0, 100, 25, "%", factor=100.0, decimals=1)
        add_humanization_row(5, "Mistake Chance:", "Gives a percentage chance for a note to be 'mispressed' (a nearby note is played for a very\nshort duration) before the correct note sounds, simulating an accidental finger slip.", 0, 10, 0, "%", factor=100.0, decimals=1)
        add_humanization_row(6, "Tempo Sway:", "Simulates expressive tempo changes (rubato) over musical phrases.\nThis value is the maximum time shift in seconds. The actual sway is randomized per phrase\nand is amplified in slow sections and reduced in fast sections.", 0, 0.1, 0, " s", factor=10000.0)

        self.vary_velocity_check = QCheckBox("Vary note velocity")
        velocity_info = self._create_info_icon("Randomly adjusts the velocity (how 'hard' a key is pressed) of each note, making the performance sound more dynamic and less robotic.")
        grid.addWidget(self.vary_velocity_check, 7, 0); grid.addWidget(velocity_info, 7, 1)
        self.humanization_sub_checkboxes.append(self.vary_velocity_check)

        self.enable_chord_roll_check = QCheckBox("Enable chord rolling")
        roll_info = self._create_info_icon("Simulates the 'rolling' of chords, where the notes are played in very quick succession from bottom to top instead of at the exact same moment.")
        grid.addWidget(self.enable_chord_roll_check, 8, 0); grid.addWidget(roll_info, 8, 1)
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
        self.use_88_key_check.setChecked(False)
        self.countdown_check.setChecked(True)
        self.debug_check.setChecked(False)

    def _reset_humanization_group_to_default(self):
        self.simulate_hands_check.setChecked(False)
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
            self.file_path_label.setText(os.path.basename(filepath))
            self.file_path_label.setToolTip(filepath)
            self.add_log_message(f"Selected file: {filepath}")

    def add_log_message(self, message): self.log_output.append(message)
    def update_progress(self, value): self.progress_bar.setValue(value)

    def clear_log(self):
        self.log_output.clear()

    def copy_log_to_clipboard(self):
        clipboard = QGuiApplication.clipboard()
        clipboard.setText(self.log_output.toPlainText())
        self.add_log_message("\n--- Log copied to clipboard. ---")

    def set_controls_enabled(self, enabled):
        self.play_button.setEnabled(enabled); self.stop_button.setEnabled(not enabled)
        self.reset_button.setEnabled(enabled)
        for groupbox in self.findChildren(QGroupBox): groupbox.setEnabled(enabled)

    def gather_config(self) -> Optional[Dict[str, Any]]:
        filepath = self.file_path_label.toolTip()
        if not filepath:
            QMessageBox.warning(self, "No File", "Please select a MIDI file before playing."); return None
        return {
            'midi_file': filepath, 'tempo': self.tempo_spinbox.value(), 'countdown': self.countdown_check.isChecked(),
            'use_88_key_layout': self.use_88_key_check.isChecked(),
            'pedal_style': self.pedal_style_combo.currentText(),
            'debug_mode': self.debug_check.isChecked(),
            # Humanization
            'simulate_hands': self.humanization_sub_checkboxes[0].isChecked(),
            'vary_timing': self.humanization_sub_checkboxes[1].isChecked(), 'timing_variance': self.humanization_spinboxes[0].value(),
            'vary_articulation': self.humanization_sub_checkboxes[2].isChecked(), 'articulation': self.humanization_spinboxes[1].value() / 100.0,
            'enable_drift_correction': self.humanization_sub_checkboxes[3].isChecked(), 'drift_decay_factor': self.humanization_spinboxes[2].value() / 100.0,
            'enable_mistakes': self.humanization_sub_checkboxes[4].isChecked(), 'mistake_chance': self.humanization_spinboxes[3].value(),
            'enable_tempo_sway': self.humanization_sub_checkboxes[5].isChecked(), 'tempo_sway_intensity': self.humanization_spinboxes[4].value(),
            'vary_velocity': self.humanization_sub_checkboxes[6].isChecked(),
            'enable_chord_roll': self.humanization_sub_checkboxes[7].isChecked(),
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
