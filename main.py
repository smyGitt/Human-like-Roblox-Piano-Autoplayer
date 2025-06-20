#!/usr/bin/env python3
"""
MIDI-to-Keyboard Player - Fully Humanized Version with Note Overlap Fix
Maps MIDI notes to computer keyboard keys with comprehensive human-like playing simulation.
Includes spacebar pedal control and extensive debug capabilities.
FIXED: Note overlap handling to prevent silent notes in external applications.
"""

import argparse
import mido
import time
import threading
import sys
import heapq
import signal
import random
import math
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional
from pynput.keyboard import Key, Controller

@dataclass
class NoteEvent:
    """MIDI note event with comprehensive timing and musical analysis data."""
    midi_note: int                    # MIDI note number (0-127)
    start_time: float                 # Note start time in seconds
    duration: float                   # Note duration in seconds
    velocity: int                     # MIDI velocity (1-127)
    original_velocity: int = None     # Unmodified velocity for reference
    is_melody: bool = False           # True if this note is part of main melody line
    chord_position: int = 0           # Position within chord (0=lowest, etc.)
    beat_strength: float = 1.0        # Rhythmic emphasis factor (0.5-1.5)
    phrase_position: str = "middle"   # Position in musical phrase: start/middle/end
    
    def __post_init__(self):
        """Initialize derived values after creation."""
        if self.original_velocity is None:
            self.original_velocity = self.velocity
    
    @property
    def end_time(self) -> float:
        """Calculate note end time from start + duration."""
        return self.start_time + self.duration

@dataclass
class ChordGroup:
    """Group of notes played simultaneously with musical analysis."""
    start_time: float                 # When chord starts playing
    left_hand_notes: List[NoteEvent]  # Notes assigned to left hand
    right_hand_notes: List[NoteEvent] # Notes assigned to right hand
    beat_position: float = 0.0        # Position within measure (0.0-1.0)
    is_strong_beat: bool = False      # True if on downbeat or strong beat
    chord_complexity: float = 1.0     # Difficulty factor (0.0-1.0)
    harmonic_tension: float = 0.5     # Dissonance level (0.0=consonant, 1.0=dissonant)
    
    @property
    def all_notes(self) -> List[NoteEvent]:
        """Get all notes in both hands combined."""
        return self.left_hand_notes + self.right_hand_notes

@dataclass
class ScheduledEvent:
    """Keyboard event scheduled for future execution with priority queue support."""
    time: float          # When to execute event (seconds from start)
    event_type: str      # Event type: 'press', 'release', or 'pedal'
    key: str             # Keyboard key to press/release
    midi_note: int       # Original MIDI note number
    hand: str            # Hand assignment: 'left', 'right', or 'unknown'
    velocity: int = 64   # Velocity for this specific event
    finger: int = 0      # Finger assignment (0=thumb, 4=pinky)
    
    def __lt__(self, other):
        """Priority queue comparison - earlier times have higher priority."""
        return self.time < other.time

@dataclass
class DebugEvent:
    """Debug information for a single playback event."""
    timestamp: float     # When event occurred
    event_type: str      # Type of event
    hand: str           # Which hand
    key: str            # Keyboard key
    midi_note: int      # MIDI note
    velocity: int       # Velocity used
    timing_adj: float   # Timing adjustment applied
    humanization: str   # Description of humanization applied

class MusicalAnalyzer:
    """Analyzes MIDI content for intelligent humanization decisions."""
    
    # Major scale patterns for key detection - maps root note to scale degrees
    MAJOR_SCALES = {
        0: [0, 2, 4, 5, 7, 9, 11],   # C major: C D E F G A B
        1: [1, 3, 5, 6, 8, 10, 0],   # C# major
        2: [2, 4, 6, 7, 9, 11, 1],   # D major
        3: [3, 5, 7, 8, 10, 0, 2],   # D# major
        4: [4, 6, 8, 9, 11, 1, 3],   # E major
        5: [5, 7, 9, 10, 0, 2, 4],   # F major
        6: [6, 8, 10, 11, 1, 3, 5],  # F# major
        7: [7, 9, 11, 0, 2, 4, 6],   # G major
        8: [8, 10, 0, 1, 3, 5, 7],   # G# major
        9: [9, 11, 1, 2, 4, 6, 8],   # A major
        10: [10, 0, 2, 3, 5, 7, 9],  # A# major
        11: [11, 1, 3, 4, 6, 8, 10]  # B major
    }
    
    @staticmethod
    def detect_key_signature(notes: List[NoteEvent]) -> int:
        """Detect most likely key signature by analyzing note frequency distribution."""
        # Count occurrences of each chromatic note
        note_counts = [0] * 12
        for note in notes:
            note_counts[note.midi_note % 12] += 1
        
        # Test each possible key and score based on scale note frequency
        best_score = -1
        best_key = 0
        
        for key, scale in MusicalAnalyzer.MAJOR_SCALES.items():
            # Calculate score as sum of scale note frequencies
            score = sum(note_counts[note % 12] for note in scale)
            if score > best_score:
                best_score = score
                best_key = key
        
        return best_key
    
    @staticmethod
    def analyze_harmony(chord_notes: List[int]) -> float:
        """Calculate harmonic tension of chord - 0.0=consonant, 1.0=dissonant."""
        if len(chord_notes) < 2:
            return 0.0
        
        # Calculate all intervals between chord notes
        intervals = []
        for i in range(len(chord_notes)):
            for j in range(i + 1, len(chord_notes)):
                interval = abs(chord_notes[i] - chord_notes[j]) % 12
                intervals.append(interval)
        
        # Consonance ratings for each interval type
        consonance = {
            0: 1.0,   # Unison - perfect consonance
            1: 0.1,   # Minor 2nd - very dissonant
            2: 0.3,   # Major 2nd - mild dissonance
            3: 0.6,   # Minor 3rd - mild consonance
            4: 0.7,   # Major 3rd - consonant
            5: 0.4,   # Perfect 4th - moderate
            6: 0.2,   # Tritone - very dissonant
            7: 0.8,   # Perfect 5th - very consonant
            8: 0.7,   # Minor 6th - consonant
            9: 0.6,   # Major 6th - mild consonance
            10: 0.3,  # Minor 7th - mild dissonance
            11: 0.1   # Major 7th - very dissonant
        }
        
        # Average consonance across all intervals, convert to tension
        avg_consonance = sum(consonance[interval] for interval in intervals) / len(intervals)
        return 1.0 - avg_consonance
    
    @staticmethod
    def detect_melody_line(notes: List[NoteEvent]) -> List[NoteEvent]:
        """Identify main melody by finding highest voice at each time point."""
        melody_notes = []
        time_groups = {}
        
        # Group notes by quantized start time
        for note in notes:
            time_key = round(note.start_time, 3)  # 1ms precision
            if time_key not in time_groups:
                time_groups[time_key] = []
            time_groups[time_key].append(note)
        
        # At each time point, mark highest note as melody
        for time_key, group in time_groups.items():
            highest = max(group, key=lambda n: n.midi_note)
            highest.is_melody = True
            melody_notes.append(highest)
        
        return melody_notes
    
    @staticmethod
    def analyze_phrases(notes: List[NoteEvent], time_signature: Tuple[int, int] = (4, 4)) -> List[NoteEvent]:
        """Analyze musical phrase structure and mark phrase boundaries."""
        if not notes:
            return notes
        
        # Sort notes chronologically for phrase analysis
        sorted_notes = sorted(notes, key=lambda n: n.start_time)
        
        # Calculate phrase length - typically 8 beats in common practice
        beats_per_phrase = 8
        beat_length = 0.5  # Assume quarter note = 500ms
        phrase_length = beats_per_phrase * beat_length
        
        # Assign phrase positions based on timing within phrase cycle
        for note in sorted_notes:
            phrase_pos = (note.start_time % phrase_length) / phrase_length
            
            if phrase_pos < 0.2:        # First 20% of phrase
                note.phrase_position = "start"
            elif phrase_pos > 0.8:      # Last 20% of phrase
                note.phrase_position = "end"
            else:                       # Middle 60% of phrase
                note.phrase_position = "middle"
        
        return sorted_notes

class MIDIKeyboardPlayer:
    """Main MIDI keyboard player with comprehensive humanization and pedal control."""
    
    # Physical keyboard layout mapped to piano keys (white + black keys interleaved)
    KEYBOARD_LAYOUT = "1!2@34$5%6^78*9(0qQwWeErtTyYuiIoOpPasSdDfgGhHjJklLzZxcCvVbBnm"
    C4_INDEX = 24           # Position of middle C in keyboard layout
    C4_MIDI = 60           # MIDI note number for middle C
    HAND_SPLIT_NOTE = 60   # Notes below this go to left hand, above to right
    
    # Finger strength simulation - realistic strength differences between fingers
    FINGER_STRENGTH = [1.0, 0.95, 0.9, 0.8, 0.7]  # thumb to pinky
    
    # Groove patterns for different musical styles - timing multipliers per beat
    SWING_PATTERNS = {
        'straight': [1.0, 1.0, 1.0, 1.0],          # Even timing
        'swing': [1.0, 0.67, 1.0, 0.67],           # Jazz swing - long-short pattern
        'shuffle': [1.0, 0.75, 1.0, 0.75],         # Blues shuffle
        'latin': [1.0, 0.9, 1.1, 0.95]             # Latin groove with slight push/pull
    }
    
    def __init__(self, tempo_scale=1.0, natural_timing=False, random_timing=False, 
                 chord_aware=False, max_timing_variance=0.05, genre=None,
                 expression_level=0.5, fatigue_simulation=False, mistake_rate=0.0,
                 debug_mode=False):
        """Initialize player with humanization parameters and setup keyboard controller."""
        self.keyboard = Controller()
        
        # Core playback parameters
        self.tempo_scale = tempo_scale                    # Global tempo multiplier
        self.natural_timing = natural_timing              # Enable natural human timing variations
        self.random_timing = random_timing                # Enable random timing/order variations
        self.chord_aware = chord_aware                    # Enable intelligent chord analysis
        self.max_timing_variance = max_timing_variance    # Maximum random timing deviation
        self.genre = genre                                # Musical style for specialized humanization
        self.expression_level = expression_level          # Emotional intensity (0.0-1.0)
        self.fatigue_simulation = fatigue_simulation      # Enable performance degradation over time
        self.mistake_rate = mistake_rate                  # Probability of simulated mistakes
        self.debug_mode = debug_mode                      # Enable detailed debug output
        
        # Musical analysis state
        self.key_signature = 0                # Detected key (0=C, 1=C#, etc.)
        self.time_signature = (4, 4)         # Time signature (numerator, denominator)
        self.current_tempo = 120.0            # Detected tempo in BPM
        self.playing_time = 0.0               # Elapsed playing time for fatigue calculation
        self.fatigue_factor = 1.0             # Current fatigue multiplier (1.0=fresh, 0.7=tired)
        
        # Performance tracking for realistic simulation
        self.performance_quality = 1.0       # Overall performance quality factor
        self.last_chord_time = 0.0           # Time of last chord for spacing calculation
        self.phrase_dynamics = []            # Dynamic curve for current phrase
        
        # MIDI to keyboard mapping setup
        self.base_midi_note = self.C4_MIDI - self.C4_INDEX
        self.midi_to_key = self._create_midi_mapping()
        
        # Thread synchronization and state management
        self.lock = threading.Lock()                      # Thread safety for shared state
        self.stop_flag = threading.Event()               # Signal to stop all threads
        self.currently_pressed_keys: Set[str] = set()    # Track active keys for display/cleanup
        self.start_time = 0                               # Performance start time reference
        self.running = False                              # Global running state flag
        self.pedal_pressed = False                        # Sustain pedal state
        
        # Event scheduling system
        self.event_queue = []                 # Priority queue for scheduled events
        self.scheduler_thread = None          # Thread for event execution
        self.display_thread = None            # Thread for real-time display
        self.last_displayed_keys = set()      # Last displayed key set for change detection
        
        # Debug tracking
        self.debug_history = []               # Complete history of debug events
        self.current_left_hand = set()        # Currently pressed left hand keys
        self.current_right_hand = set()       # Currently pressed right hand keys
        
        self._setup_signal_handlers()
        
    def _setup_signal_handlers(self):
        """Configure signal handlers for graceful shutdown on Ctrl+C or system termination."""
        def signal_handler(signum, frame):
            self.stop_flag.set()
            self.running = False
            
        signal.signal(signal.SIGINT, signal_handler)
        if hasattr(signal, 'SIGTERM'):
            signal.signal(signal.SIGTERM, signal_handler)
    
    def _create_midi_mapping(self) -> Dict[int, str]:
        """Create bidirectional mapping from MIDI note numbers to keyboard characters."""
        mapping = {}
        for i, key in enumerate(self.KEYBOARD_LAYOUT):
            midi_note = self.base_midi_note + i
            mapping[midi_note] = key
        return mapping
    
    def _map_note_to_key(self, midi_note: int) -> Optional[str]:
        """Map MIDI note to keyboard key, auto-transposing out-of-range notes to nearest octave."""
        if midi_note in self.midi_to_key:
            return self.midi_to_key[midi_note]
        
        # Transpose by octaves until note fits in available range
        while midi_note < self.base_midi_note:
            midi_note += 12
        while midi_note >= self.base_midi_note + len(self.KEYBOARD_LAYOUT):
            midi_note -= 12
            
        return self.midi_to_key.get(midi_note)
    
    def _assign_finger(self, midi_note: int, hand: str, hand_notes: List[NoteEvent]) -> int:
        """Assign realistic finger numbers based on note position within hand."""
        if hand == 'left':
            # Left hand: thumb plays highest notes, pinky plays lowest
            sorted_notes = sorted(hand_notes, key=lambda n: n.midi_note, reverse=True)
        else:
            # Right hand: thumb plays lowest notes, pinky plays highest
            sorted_notes = sorted(hand_notes, key=lambda n: n.midi_note)
        
        try:
            # Find position of this note in sorted order
            position = next(i for i, note in enumerate(sorted_notes) if note.midi_note == midi_note)
            return min(position, 4)  # Clamp to valid finger range (0-4)
        except StopIteration:
            return 0  # Default to thumb if note not found
    
    def _get_beat_strength(self, time_pos: float, time_signature: Tuple[int, int] = (4, 4)) -> float:
        """Calculate rhythmic emphasis based on position within measure - strong beats get emphasis."""
        beats_per_measure = time_signature[0]
        beat_length = 0.5  # Assume quarter note = 500ms for beat calculation
        
        beat_position = (time_pos / beat_length) % beats_per_measure
        
        # Assign strength based on traditional rhythmic hierarchy
        if beat_position < 0.1:              # Downbeat (beat 1)
            return 1.2
        elif beat_position % 1.0 < 0.1:      # Other strong beats
            return 1.0
        elif beat_position % 0.5 < 0.1:      # Off-beats (syncopation)
            return 0.9
        else:                                # Weak subdivisions
            return 0.8
    
    def _apply_swing(self, time_pos: float, genre: str = 'straight') -> float:
        """Apply genre-specific timing adjustments for musical groove."""
        if genre not in self.SWING_PATTERNS:
            return time_pos
        
        pattern = self.SWING_PATTERNS[genre]
        beat_length = 0.5
        beat_position = (time_pos / beat_length) % len(pattern)
        beat_index = int(beat_position)
        
        # Apply swing factor as timing adjustment
        swing_factor = pattern[beat_index]
        adjustment = (swing_factor - 1.0) * 0.05  # Max 50ms swing adjustment
        
        return time_pos + adjustment
    
    def _calculate_velocity_humanization(self, note: NoteEvent, finger: int, hand: str) -> int:
        """Calculate realistic velocity based on finger strength, musical context, and humanization."""
        base_velocity = note.velocity
        
        # Physical finger strength variation - thumb strongest, pinky weakest
        finger_factor = self.FINGER_STRENGTH[finger]
        
        # Register-based dynamics - bass notes naturally heavier, treble lighter
        if note.midi_note < 48:              # Bass register (below C3)
            register_factor = 1.15
        elif note.midi_note > 72:            # Treble register (above C5)
            register_factor = 0.9
        else:                                # Middle register
            register_factor = 1.0
        
        # Emphasize melody line for musical clarity
        melody_factor = 1.1 if note.is_melody else 1.0
        
        # Apply beat emphasis for rhythmic structure
        beat_factor = note.beat_strength
        
        # Random humanization within reasonable bounds
        random_factor = random.uniform(0.85, 1.15) if self.random_timing else 1.0
        
        # Fatigue causes gradual velocity reduction over time
        fatigue_factor = self.fatigue_factor if self.fatigue_simulation else 1.0
        
        # Expression level controls overall dynamic range
        expression_range = 0.3 * self.expression_level
        expression_factor = 1.0 + random.uniform(-expression_range, expression_range)
        
        # Combine all factors multiplicatively
        final_velocity = base_velocity * finger_factor * register_factor * melody_factor
        final_velocity *= beat_factor * random_factor * fatigue_factor * expression_factor
        
        # Clamp to valid MIDI velocity range
        return max(1, min(127, int(final_velocity)))
    
    def _calculate_timing_humanization(self, note: NoteEvent, chord_notes: List[NoteEvent], 
                                     hand: str, finger: int) -> Tuple[float, float]:
        """Calculate comprehensive timing adjustments for note press and duration."""
        press_adjustment = 0.0
        duration_adjustment = 0.0
        
        # Basic random timing variation if enabled
        if self.random_timing:
            if random.random() < 0.7:  # 70% chance of having timing variation
                press_adjustment += random.uniform(-self.max_timing_variance, self.max_timing_variance)
        
        # Natural human timing characteristics
        if self.natural_timing:
            # Chord rolling - slight stagger between simultaneous notes
            if len(chord_notes) > 1:
                note_index = next((i for i, n in enumerate(chord_notes) if n.midi_note == note.midi_note), 0)
                roll_delay = note_index * 0.005  # 5ms between each note in chord
                press_adjustment += roll_delay
            
            # Musical phrase breathing - hesitation at phrase starts, lengthening at ends
            if note.phrase_position == "start":
                press_adjustment += random.uniform(0.0, 0.02)  # Up to 20ms hesitation
            elif note.phrase_position == "end":
                duration_adjustment += random.uniform(0.0, 0.05)  # Up to 50ms lengthening
        
        # Beat emphasis timing - strong beats slightly early for forward momentum
        if note.beat_strength > 1.0:
            press_adjustment -= 0.005  # Strong beats 5ms early
        elif note.beat_strength < 0.9:
            press_adjustment += 0.003  # Weak beats 3ms late
        
        # Genre-specific swing application
        if self.genre in ['jazz', 'swing', 'blues']:
            swing_adjustment = self._apply_swing(note.start_time, 'swing') - note.start_time
            press_adjustment += swing_adjustment
        
        # Finger-dependent timing - weaker fingers are slightly slower
        if finger > 2:  # Ring finger and pinky have delayed response
            press_adjustment += 0.002 * (finger - 2)
        
        # Harmonic complexity affects timing precision
        chord_midi_notes = [n.midi_note for n in chord_notes]
        tension = MusicalAnalyzer.analyze_harmony(chord_midi_notes)
        if tension > 0.7:  # High harmonic tension requires more careful timing
            press_adjustment += 0.005
        
        # Fatigue effects - tired players have less precise timing
        if self.fatigue_simulation:
            press_adjustment += (1.0 - self.fatigue_factor) * 0.01
            duration_adjustment -= (1.0 - self.fatigue_factor) * 0.02
        
        # Genre-specific timing characteristics
        if self.genre == 'classical':
            press_adjustment *= 0.7      # More precise, less variation
        elif self.genre == 'jazz':
            press_adjustment *= 1.3      # More swing and rhythmic freedom
        elif self.genre == 'romantic':
            # More rubato - emotional timing flexibility
            press_adjustment += random.uniform(-0.01, 0.01) * self.expression_level
        
        return press_adjustment, duration_adjustment
    
    def _calculate_articulation(self, note: NoteEvent, next_note: Optional[NoteEvent] = None) -> float:
        """Calculate note length based on musical articulation and context."""
        base_duration = note.duration
        
        # Default articulation - slightly detached for clarity
        articulation_factor = 0.95
        
        # Legato connection for smooth melody lines
        if note.is_melody and next_note and abs(next_note.midi_note - note.midi_note) <= 2:
            articulation_factor = 1.05  # Slight overlap for legato connection
        
        # Staccato articulation for short notes or fast passages
        if base_duration < 0.2:  # Notes shorter than 200ms
            articulation_factor = 0.8
        
        # Phrase-sensitive articulation adjustments
        if note.phrase_position == "end":
            articulation_factor *= 1.1  # Longer notes at phrase endings
        
        # Genre-specific articulation styles
        if self.genre == 'baroque':
            articulation_factor = 0.9   # More detached, historically informed
        elif self.genre == 'romantic':
            articulation_factor = 1.0   # More connected, expressive
        
        return base_duration * articulation_factor
    
    def _get_key_combination(self, key: str) -> Tuple[List, str]:
        """Convert keyboard layout character to actual key combination needed."""
        # Symbol mapping for shifted number keys
        symbol_map = {
            '!': '1', '@': '2', '#': '3', '$': '4', '%': '5',
            '^': '6', '&': '7', '*': '8', '(': '9', ')': '0'
        }
        
        if key in symbol_map:
            return ([Key.shift], symbol_map[key])  # Shift + number for symbol
        elif key.isupper():
            return ([Key.shift], key.lower())      # Shift + letter for uppercase
        else:
            return ([], key)                       # Plain key press
    
    def _simulate_mistake(self, note: NoteEvent) -> bool:
        """Simulate occasional human playing mistakes based on difficulty and fatigue."""
        if self.mistake_rate <= 0:
            return False
        
        # Base mistake probability from configuration
        mistake_probability = self.mistake_rate
        
        # Fatigue increases mistake probability
        if self.fatigue_simulation:
            mistake_probability *= (2.0 - self.fatigue_factor)
        
        # Difficult finger combinations more prone to mistakes
        if hasattr(note, 'finger') and note.finger > 3:
            mistake_probability *= 1.5
        
        return random.random() < mistake_probability
    
    def _log_debug_event(self, event_type: str, hand: str, key: str, midi_note: int, 
                        velocity: int, timing_adj: float, humanization: str):
        """Log debug event for analysis and history tracking."""
        if not self.debug_mode:
            return
            
        debug_event = DebugEvent(
            timestamp=time.perf_counter() - self.start_time,
            event_type=event_type,
            hand=hand,
            key=key,
            midi_note=midi_note,
            velocity=velocity,
            timing_adj=timing_adj,
            humanization=humanization
        )
        
        self.debug_history.append(debug_event)
        
        # Real-time debug output
        hand_state = f"L:{sorted(self.current_left_hand)} R:{sorted(self.current_right_hand)}"
        print(f"[{debug_event.timestamp:6.3f}] {event_type:7} {hand:5} {key:1} "
              f"(MIDI:{midi_note:3}) vel:{velocity:3} adj:{timing_adj:+.3f}s | {hand_state}")
    

    def _group_notes_into_chords(self, notes: List[NoteEvent]) -> List[ChordGroup]:
        """Group notes by timing and analyze musical structure for intelligent humanization."""
        if not notes:
            return []
        
        # Perform comprehensive musical analysis
        self.key_signature = MusicalAnalyzer.detect_key_signature(notes)
        melody_notes = MusicalAnalyzer.detect_melody_line(notes)
        analyzed_notes = MusicalAnalyzer.analyze_phrases(notes, self.time_signature)
        
        # Apply beat strength analysis to all notes
        for note in analyzed_notes:
            note.beat_strength = self._get_beat_strength(note.start_time, self.time_signature)
        
        # Simple mode: treat each note as individual "chord"
        if not self.chord_aware:
            chord_groups = []
            for note in analyzed_notes:
                chord = ChordGroup(note.start_time, [], [note])
                chord.beat_position = note.start_time % (self.time_signature[0] * 0.5)
                chord.is_strong_beat = note.beat_strength > 1.0
                chord_groups.append(chord)
            return chord_groups
        
        # FIXED: Chord-aware mode with millisecond quantization to prevent floating-point precision issues
        time_groups = {}
        for note in analyzed_notes:
            ms_time = int(note.start_time * 1000)  # Quantize to milliseconds
            if ms_time not in time_groups:
                time_groups[ms_time] = []
            time_groups[ms_time].append(note)
        
        chord_groups = []
        for ms_time, chord_notes in time_groups.items():
            start_time = ms_time / 1000.0  # Convert back to seconds
            
            # Analyze chord characteristics for humanization
            midi_notes = [n.midi_note for n in chord_notes]
            harmonic_tension = MusicalAnalyzer.analyze_harmony(midi_notes)
            complexity = min(len(chord_notes) / 6.0, 1.0)  # Normalize to 0-1 range
            
            # Hand assignment based on Middle C split point
            left_hand = [n for n in chord_notes if n.midi_note <= self.HAND_SPLIT_NOTE]
            right_hand = [n for n in chord_notes if n.midi_note > self.HAND_SPLIT_NOTE]
            
            # Handle hand overflow (>5 notes per hand) by reassigning notes
            if len(left_hand) > 5:
                # Move highest left-hand notes to right hand
                sorted_left = sorted(left_hand, key=lambda n: n.midi_note)
                overflow_count = len(left_hand) - 5
                overflow_notes = sorted_left[-overflow_count:]
                left_hand = sorted_left[:-overflow_count]
                right_hand.extend(overflow_notes)
            
            if len(right_hand) > 5:
                # Move lowest right-hand notes to left hand
                sorted_right = sorted(right_hand, key=lambda n: n.midi_note)
                overflow_count = len(right_hand) - 5
                overflow_notes = sorted_right[:overflow_count]
                right_hand = sorted_right[overflow_count:]
                left_hand.extend(overflow_notes)
            
            # Create analyzed chord group
            chord = ChordGroup(start_time, left_hand, right_hand)
            chord.beat_position = start_time % (self.time_signature[0] * 0.5)
            chord.is_strong_beat = any(n.beat_strength > 1.0 for n in chord_notes)
            chord.chord_complexity = complexity
            chord.harmonic_tension = harmonic_tension
            
            chord_groups.append(chord)
        
        return sorted(chord_groups, key=lambda c: c.start_time)
    
    def _apply_humanization_to_chord(self, chord: ChordGroup) -> List[Tuple[float, str, str, int, str, int]]:
        """Apply comprehensive humanization to chord group, returning scheduled events."""
        events = []
        
        # Update fatigue simulation based on elapsed playing time
        if self.fatigue_simulation:
            time_played = chord.start_time
            # Gradual fatigue over 10 minutes (600 seconds)
            self.fatigue_factor = max(0.7, 1.0 - (time_played / 600.0) * 0.3)
        
        # Process each hand separately for realistic coordination
        for hand, hand_notes in [('left', chord.left_hand_notes), ('right', chord.right_hand_notes)]:
            if not hand_notes:
                continue
            
            # Assign finger numbers based on hand position and note arrangement
            for note in hand_notes:
                note.finger = self._assign_finger(note.midi_note, hand, hand_notes)
            
            # Apply note ordering based on humanization settings
            if self.random_timing:
                # Random order for experimental/avant-garde effect
                hand_notes = hand_notes.copy()
                random.shuffle(hand_notes)
            elif self.natural_timing and len(hand_notes) > 1:
                # Natural chord rolling: bass-to-treble for left, treble-to-bass for right
                if hand == 'left':
                    hand_notes = sorted(hand_notes, key=lambda n: n.midi_note)
                else:
                    hand_notes = sorted(hand_notes, key=lambda n: n.midi_note, reverse=True)
            
            # Apply humanization to each individual note
            for i, note in enumerate(hand_notes):
                # Skip note if mistake simulation triggers
                if self._simulate_mistake(note):
                    continue
                
                # Map MIDI note to keyboard key
                key = self._map_note_to_key(note.midi_note)
                if not key:
                    continue
                
                # Calculate comprehensive timing adjustments
                press_adjustment, duration_adjustment = self._calculate_timing_humanization(
                    note, hand_notes, hand, note.finger
                )
                
                # Calculate humanized velocity based on multiple factors
                velocity = self._calculate_velocity_humanization(note, note.finger, hand)
                
                # Calculate articulation-adjusted duration
                articulated_duration = self._calculate_articulation(note)
                
                # Apply all timing and duration adjustments
                actual_press_time = max(0, note.start_time + press_adjustment)
                actual_duration = max(0.05, articulated_duration + duration_adjustment)
                actual_release_time = actual_press_time + actual_duration
                
                # Create humanization description for debug logging
                humanization_desc = f"finger:{note.finger} tension:{chord.harmonic_tension:.2f}"
                if self.fatigue_simulation:
                    humanization_desc += f" fatigue:{self.fatigue_factor:.2f}"
                
                # Log debug information if enabled
                self._log_debug_event('press', hand, key, note.midi_note, velocity, 
                                    press_adjustment, humanization_desc)
                
                # Schedule press and release events (simple approach)
                events.append((actual_press_time, 'press', key, note.midi_note, hand, velocity))
                events.append((actual_release_time, 'release', key, note.midi_note, hand, velocity))
        
        return events
    
    def _press_key(self, key: str, midi_note: int, velocity: int = 64):
        """Execute keyboard key press - allows overlapping presses."""
        if self.stop_flag.is_set():
            return
            
        with self.lock:
            # Remove the overlap prevention check - always press the key
            try:
                modifiers, base_key = self._get_key_combination(key)
                requires_shift = Key.shift in modifiers
                
                # Use separate controller instance for atomic operations
                controller = Controller()
                
                if requires_shift:
                    # Atomic shift+key operation using context manager
                    with controller.pressed(Key.shift):
                        time.sleep(0.0005)  # Micro-delay for OS processing
                        controller.press(base_key)
                        time.sleep(0.0005)  # Ensure key registers in shift context
                    # Shift automatically released by context manager
                else:
                    # Normal key press without modifiers
                    controller.press(base_key)
                    time.sleep(0.001)  # Standard delay for OS processing
                
                self.currently_pressed_keys.add(key)
                
                # Update hand tracking for debug display
                if self.debug_mode:
                    # Determine hand based on MIDI note (rough approximation)
                    if midi_note <= self.HAND_SPLIT_NOTE:
                        self.current_left_hand.add(key)
                    else:
                        self.current_right_hand.add(key)
                
            except Exception:
                pass  # Silently handle key press failures
    
    def _release_key(self, key: str, midi_note: int, velocity: int = 64):
        """Execute keyboard key release."""
        if self.stop_flag.is_set():
            return
            
        with self.lock:
            # Always attempt to release the key
            try:
                modifiers, base_key = self._get_key_combination(key)
                
                # Use main keyboard controller for release
                self.keyboard.release(base_key)
                time.sleep(0.001)  # Allow OS to process release
                
                self.currently_pressed_keys.discard(key)
                
                # Update hand tracking for debug display
                if self.debug_mode:
                    self.current_left_hand.discard(key)
                    self.current_right_hand.discard(key)
                    self._log_debug_event('release', 'both', key, midi_note, velocity, 0, 'normal')
                
            except Exception:
                pass  # Silently handle key release failures
    
    def _handle_pedal(self, pedal_state: bool):
        """Handle sustain pedal press/release using spacebar."""
        if self.stop_flag.is_set():
            return
            
        try:
            if pedal_state and not self.pedal_pressed:
                # Press pedal (spacebar down)
                self.keyboard.press(Key.space)
                self.pedal_pressed = True
                if self.debug_mode:
                    self._log_debug_event('pedal', 'both', 'SPACE', 0, 127, 0, 'sustain_on')
            elif not pedal_state and self.pedal_pressed:
                # Release pedal (spacebar up)
                self.keyboard.release(Key.space)
                self.pedal_pressed = False
                if self.debug_mode:
                    self._log_debug_event('pedal', 'both', 'SPACE', 0, 0, 0, 'sustain_off')
        except Exception:
            pass  # Silently handle pedal failures
    
    def _schedule_event(self, event_time: float, event_type: str, key: str, 
                       midi_note: int, hand: str = 'unknown', velocity: int = 64):
        """Add event to priority queue for future execution."""
        if event_type == 'pedal':
            # Special handling for pedal events
            event = ScheduledEvent(event_time, event_type, key, midi_note, hand, velocity)
        else:
            # Normal note events
            event = ScheduledEvent(event_time, event_type, key, midi_note, hand, velocity)
        heapq.heappush(self.event_queue, event)
    
    def _schedule_chord(self, chord: ChordGroup):
        """Schedule entire chord with comprehensive humanization applied."""
        humanized_events = self._apply_humanization_to_chord(chord)
        
        # Add all humanized events to the scheduler queue
        for event_time, event_type, key, midi_note, hand, velocity in humanized_events:
            self._schedule_event(event_time, event_type, key, midi_note, hand, velocity)
        
        # Add basic pedal simulation for longer chords (>1 second)
        if chord.all_notes and max(note.duration for note in chord.all_notes) > 1.0:
            # Press pedal slightly before chord
            self._schedule_event(chord.start_time - 0.01, 'pedal', 'SPACE', 0, 'both', 127)
            # Release pedal after longest note
            max_end_time = max(note.end_time for note in chord.all_notes)
            self._schedule_event(max_end_time + 0.1, 'pedal', 'SPACE', 0, 'both', 0)
    
    def _scheduler_loop(self):
        """Main event scheduler loop with timing compensation for consistent playback."""
        start_perf = time.perf_counter()
        accumulated_delay = 0.0
        
        while self.running and not self.stop_flag.is_set():
            current_time = time.perf_counter() - start_perf
            
            # Check if any events are ready for execution
            if self.event_queue and self.event_queue[0].time <= (current_time + accumulated_delay):
                event = heapq.heappop(self.event_queue)
                
                operation_start = time.perf_counter()
                
                # Execute event based on type
                if event.event_type == 'press':
                    self._press_key(event.key, event.midi_note, event.velocity)
                elif event.event_type == 'release':
                    self._release_key(event.key, event.midi_note, event.velocity)
                elif event.event_type == 'pedal':
                    pedal_state = event.velocity > 0  # velocity > 0 = press, 0 = release
                    self._handle_pedal(pedal_state)
                
                # Track operation time for timing compensation
                operation_time = time.perf_counter() - operation_start
                accumulated_delay += operation_time
            else:
                # No events ready - brief sleep to prevent CPU spinning
                time.sleep(0.001)
    
    def _display_current_keys(self):
        """Display current key state - minimal for normal mode, detailed for debug mode."""
        while self.running and not self.stop_flag.is_set():
            with self.lock:
                current_keys = self.currently_pressed_keys.copy()
            
            # Only update display when keys change to reduce flicker
            if current_keys != self.last_displayed_keys:
                sys.stdout.write('\r' + ' ' * 120 + '\r')
                
                if self.debug_mode:
                    # Debug mode: show detailed hand information
                    left_display = ' '.join(sorted(self.current_left_hand))
                    right_display = ' '.join(sorted(self.current_right_hand))
                    
                    if current_keys:
                        pedal_indicator = " [PEDAL]" if self.pedal_pressed else ""
                        genre_tag = f"[{self.genre.upper()}]" if self.genre else "[NO GENRE]"
                        fatigue_info = f" fatigue:{self.fatigue_factor:.2f}" if self.fatigue_simulation else ""
                        
                        display = f"🎹 {genre_tag} L:[{left_display}] R:[{right_display}]{pedal_indicator}{fatigue_info}"
                        sys.stdout.write(display)
                    else:
                        genre_tag = f"[{self.genre.upper()}]" if self.genre else "[NO GENRE]"
                        sys.stdout.write(f"🎹 {genre_tag} Ready")
                else:
                    # Normal mode: minimal display (only for debug flag)
                    pass  # No output in normal mode
                
                sys.stdout.flush()
                self.last_displayed_keys = current_keys
            
            time.sleep(0.05)  # 20Hz update rate
    
    def load_midi_file(self, filename: str) -> List[ChordGroup]:
        """Load MIDI file and parse into chord groups with comprehensive musical analysis."""
        try:
            mid = mido.MidiFile(filename)
        except Exception as e:
            raise ValueError(f"Error loading MIDI file: {e}")
        
        notes = []
        current_time = 0
        active_notes = {}
        
        # Extract musical metadata from MIDI file
        for track in mid.tracks:
            for msg in track:
                if msg.type == 'set_tempo':
                    self.current_tempo = mido.tempo2bpm(msg.tempo)
                elif msg.type == 'time_signature':
                    self.time_signature = (msg.numerator, msg.denominator)
        
        # Process all MIDI note events into NoteEvent objects
        for msg in mid:
            current_time += msg.time / self.tempo_scale  # Fixed: divide instead of multiply
            
            if msg.type == 'note_on' and msg.velocity > 0:
                # Note start - store in active notes dictionary
                active_notes[msg.note] = {
                    'start_time': current_time,
                    'velocity': msg.velocity
                }
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                # Note end - create NoteEvent if note was active
                if msg.note in active_notes:
                    note_data = active_notes.pop(msg.note)
                    duration = current_time - note_data['start_time']
                    
                    note_event = NoteEvent(
                        midi_note=msg.note,
                        start_time=note_data['start_time'],
                        duration=duration,
                        velocity=note_data['velocity']
                    )
                    notes.append(note_event)
        
        # Group notes into chords with full musical analysis
        chord_groups = self._group_notes_into_chords(notes)
        return chord_groups
    
    def _get_user_confirmation(self) -> bool:
        """Get user confirmation before starting playback."""
        while True:
            try:
                response = input("Start playback? (y/n): ").strip().lower()
                if response in ['y', 'yes']:
                    return True
                elif response in ['n', 'no']:
                    return False
                else:
                    print("Please enter 'y' or 'n'")
            except (EOFError, KeyboardInterrupt):
                return False
    
    def countdown(self, seconds=3):
        """Display countdown before starting playback."""
        print(f"Starting in:", end="")
        for i in range(seconds, 0, -1):
            if self.stop_flag.is_set():
                return
            print(f" {i}", end="", flush=True)
            time.sleep(1)
        print(" GO!")
    
    def _cleanup_keys(self):
        """Release all currently pressed keys and cleanup stuck modifiers."""
        with self.lock:
            keys_to_release = list(self.currently_pressed_keys)
        
        # Release all active keys
        for key in keys_to_release:
            try:
                modifiers, base_key = self._get_key_combination(key)
                self.keyboard.release(base_key)
            except Exception:
                pass
        
        # Force cleanup any stuck modifiers and pedal
        try:
            self.keyboard.release(Key.shift)
            self.keyboard.release(Key.ctrl)
            self.keyboard.release(Key.alt)
            self.keyboard.release(Key.space)  # Release pedal
        except Exception:
            pass
        
        with self.lock:
            self.currently_pressed_keys.clear()
            if self.debug_mode:
                self.current_left_hand.clear()
                self.current_right_hand.clear()
        
        self.pedal_pressed = False
    
    def _cleanup_threads(self, timeout=2):
        """Clean up threads with aggressive termination and timeout handling."""
        if self.debug_mode:
            print("\nStopping threads...")
        
        # Set stop flags first
        self.running = False
        self.stop_flag.set()
        
        # Give threads a moment to see the stop flags
        time.sleep(0.1)
        
        # Try to join scheduler thread with timeout
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=timeout)
            if self.scheduler_thread.is_alive() and self.debug_mode:
                print("Warning: Scheduler thread did not stop cleanly")
        
        # Try to join display thread with timeout
        if self.display_thread and self.display_thread.is_alive():
            self.display_thread.join(timeout=timeout)
            if self.display_thread.is_alive() and self.debug_mode:
                print("Warning: Display thread did not stop cleanly")
        
        # Clear the display line
        sys.stdout.write('\r' + ' ' * 120 + '\r')
        sys.stdout.flush()
    
    def play_midi_file(self, filename: str, confirmation_enabled=True, countdown_enabled=True):
        """Main playback function with comprehensive setup and execution."""
        # Load and analyze MIDI file
        chord_groups = self.load_midi_file(filename)
        
        if not chord_groups:
            print("No notes found!")
            return
        
        # Display comprehensive file analysis
        total_notes = sum(len(chord.all_notes) for chord in chord_groups)
        print(f"Loaded {total_notes} notes in {len(chord_groups)} chord groups")
        print(f"Key signature: {self.key_signature} | Time signature: {self.time_signature[0]}/{self.time_signature[1]}")
        print(f"Tempo: {self.current_tempo:.1f} BPM | Genre: {self.genre or 'None'}")
        
        # Show hand distribution for chord-aware mode
        if self.chord_aware:
            left_hand_chords = sum(1 for chord in chord_groups if chord.left_hand_notes)
            right_hand_chords = sum(1 for chord in chord_groups if chord.right_hand_notes)
            print(f"Hand distribution: {left_hand_chords} left, {right_hand_chords} right chord groups")
        
        # Display active humanization features
        features = []
        if self.natural_timing: features.append("natural timing")
        if self.random_timing: features.append("random timing")
        if self.fatigue_simulation: features.append("fatigue simulation")
        if self.mistake_rate > 0: features.append(f"mistakes ({self.mistake_rate:.1%})")
        if self.expression_level > 0: features.append(f"expression ({self.expression_level:.1f})")
        if self.debug_mode: features.append("debug mode")
        
        if features:
            print(f"Humanization: {', '.join(features)}")
        
        # User confirmation step
        if confirmation_enabled and not self._get_user_confirmation():
            return
        
        # Countdown before playback
        if countdown_enabled:
            self.countdown()
            if self.stop_flag.is_set():
                return
        
        # Initialize playback state
        self.running = True
        self.start_time = time.perf_counter()
        
        # Start background threads
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop)
        self.display_thread = threading.Thread(target=self._display_current_keys)
        
        self.scheduler_thread.start()
        self.display_thread.start()
        
        # Schedule all chord groups for playback
        for chord in chord_groups:
            if self.stop_flag.is_set():
                break
            self._schedule_chord(chord)
        
        # Wait for playback completion
        if chord_groups and not self.stop_flag.is_set():
            # Calculate total duration including all note endings
            total_duration = max(note.end_time for chord in chord_groups for note in chord.all_notes)
            end_time = time.perf_counter() + total_duration + 2  # Extra 2 seconds for cleanup
            
            while time.perf_counter() < end_time and not self.stop_flag.is_set():
                time.sleep(0.1)
        
        # Force cleanup regardless of how we got here
        self.running = False
        self.stop_flag.set()
        
        try:
            self._cleanup_threads()
            self._cleanup_keys()
        except Exception as e:
            if self.debug_mode:
                print(f"Cleanup error: {e}")
        
        # Final status message
        if self.stop_flag.is_set():
            print("Playback interrupted!")
        else:
            print("Done!")
        
        # Debug summary
        if self.debug_mode and self.debug_history:
            print(f"\nDebug Summary: {len(self.debug_history)} events logged")
        
        # Ensure clean output
        sys.stdout.write('\n')
        sys.stdout.flush()

def main():
    """Main function with comprehensive command-line interface and argument validation."""
    parser = argparse.ArgumentParser(description="MIDI-to-Keyboard Player with Full Humanization and Pedal Control")
    
    # Required argument
    parser.add_argument('midi_file', help='MIDI file to play')
    
    # Basic playback parameters
    parser.add_argument('--tempo', type=float, default=1.0, help='Tempo scaling factor: 2.0=2x faster, 0.5=2x slower (default: 1.0)')
    
    # Core humanization flags
    parser.add_argument('-n', '--natural', action='store_true', 
                       help='Enable natural timing variations and chord rolling')
    parser.add_argument('-r', '--random-timing', action='store_true',
                       help='Enable random timing variations and note order')
    parser.add_argument('-c', '--chord-aware', action='store_true',
                       help='Enable chord grouping and hand assignment')
    
    # Advanced humanization parameters
    parser.add_argument('--genre', choices=['classical', 'jazz', 'romantic', 'baroque', 'blues', 'pop'], 
                       help='Musical genre for style-specific humanization (default: none)')
    parser.add_argument('--expression', type=float, default=0.5, 
                       help='Expression level 0.0-1.0 (default: 0.5)')
    parser.add_argument('--fatigue', action='store_true', 
                       help='Enable fatigue simulation over time')
    parser.add_argument('--mistakes', type=float, default=0.0, 
                       help='Mistake rate 0.0-0.1 (default: 0.0)')
    
    # Timing fine-tuning
    parser.add_argument('--timing-variance', type=float, default=0.05,
                       help='Maximum timing variance in seconds (default: 0.05)')
    
    # Interface and debug options
    parser.add_argument('--debug', action='store_true', 
                       help='Enable detailed debug output with playback history')
    parser.add_argument('--no-confirmation', action='store_true', help='Skip confirmation prompt')
    parser.add_argument('--no-countdown', action='store_true', help='Skip countdown before playback')
    
    args = parser.parse_args()
    
    # Validate all numeric parameters
    if args.timing_variance < 0 or args.timing_variance > 0.5:
        print("Error: Timing variance must be between 0.0 and 0.5 seconds")
        sys.exit(1)
    
    if args.tempo <= 0:
        print("Error: Tempo must be greater than 0")
        sys.exit(1)
    
    if args.expression < 0 or args.expression > 1:
        print("Error: Expression level must be between 0.0 and 1.0")
        sys.exit(1)
    
    if args.mistakes < 0 or args.mistakes > 0.1:
        print("Error: Mistake rate must be between 0.0 and 0.1")
        sys.exit(1)
    
    # Create player with all specified parameters
    player = MIDIKeyboardPlayer(
        tempo_scale=args.tempo,
        natural_timing=args.natural,
        random_timing=args.random_timing,
        chord_aware=args.chord_aware,
        max_timing_variance=args.timing_variance,
        genre=args.genre,  # Can be None if not specified
        expression_level=args.expression,
        fatigue_simulation=args.fatigue,
        mistake_rate=args.mistakes,
        debug_mode=args.debug
    )
    
    # Display comprehensive startup information
    features = []
    if args.natural: features.append("natural timing")
    if args.random_timing: features.append("random timing")
    if args.chord_aware: features.append("chord-aware")
    if args.fatigue: features.append("fatigue simulation")
    if args.mistakes > 0: features.append(f"mistakes ({args.mistakes:.1%})")
    if args.expression > 0: features.append(f"expression ({args.expression:.1f})")
    if args.debug: features.append("debug mode")
    
    print(f"🎹 Enhanced MIDI Player with Pedal Control")
    print(f"Genre: {args.genre.title() if args.genre else 'None'} | Tempo: {args.tempo}x")
    if features:
        print(f"Features: {', '.join(features)}")
    if args.debug:
        print("Debug mode: Detailed output enabled")
    
    # Execute main playback with comprehensive error handling
    try:
        player.play_midi_file(
            args.midi_file, 
            confirmation_enabled=not args.no_confirmation,
            countdown_enabled=not args.no_countdown
        )
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        player.stop_flag.set()
        player.running = False
    except Exception as e:
        print(f"Error: {e}")
        player.stop_flag.set()
        player.running = False
    finally:
        # Ensure absolutely clean exit
        try:
            player.stop_flag.set()
            player.running = False
            time.sleep(0.1)  # Give threads time to stop gracefully
        except:
            pass
        
        # Force exit if still hanging
        sys.exit(0)

if __name__ == "__main__":
    main()
