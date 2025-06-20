#!/usr/bin/env python3
"""
MIDI-to-Keyboard Player - Enhanced Version with Chord Grouping
Maps MIDI notes to computer keyboard keys with piano-like hand assignment.
"""

import argparse
import mido
import time
import threading
import sys
import heapq
import signal
import random
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional
from pynput.keyboard import Key, Controller

@dataclass
class NoteEvent:
    """Represents a MIDI note event with timing information."""
    midi_note: int
    start_time: float
    duration: float
    velocity: int
    
    @property
    def end_time(self) -> float:
        return self.start_time + self.duration

@dataclass
class ChordGroup:
    """Represents a group of notes played simultaneously."""
    start_time: float
    left_hand_notes: List[NoteEvent]
    right_hand_notes: List[NoteEvent]
    
    @property
    def all_notes(self) -> List[NoteEvent]:
        return self.left_hand_notes + self.right_hand_notes

@dataclass
class ScheduledEvent:
    """Represents a scheduled keyboard event."""
    time: float
    event_type: str  # 'press' or 'release'
    key: str
    midi_note: int
    hand: str  # 'left', 'right', or 'unknown'
    
    def __lt__(self, other):
        return self.time < other.time

class MIDIKeyboardPlayer:
    """Main MIDI keyboard player class."""
    
    KEYBOARD_LAYOUT = "1!2@34$5%6^78*9(0qQwWeErtTyYuiIoOpPasSdDfgGhHjJklLzZxcCvVbBnm"
    C4_INDEX = 24
    C4_MIDI = 60
    HAND_SPLIT_NOTE = 60  # Middle C
    
    def __init__(self, tempo_scale=1.0, natural_timing=False, random_timing=False, 
                 chord_aware=False, max_timing_variance=0.05):
        self.keyboard = Controller()
        self.tempo_scale = tempo_scale
        self.natural_timing = natural_timing
        self.random_timing = random_timing
        self.chord_aware = chord_aware
        self.max_timing_variance = max_timing_variance
        
        # MIDI to keyboard mapping
        self.base_midi_note = self.C4_MIDI - self.C4_INDEX
        self.midi_to_key = self._create_midi_mapping()
        
        # Thread synchronization
        self.lock = threading.Lock()
        self.stop_flag = threading.Event()
        self.currently_pressed_keys: Set[str] = set()
        self.start_time = 0
        self.running = False
        
        # Scheduler
        self.event_queue = []
        self.scheduler_thread = None
        self.display_thread = None
        self.last_displayed_keys = set()
        
        self._setup_signal_handlers()
        
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            self.stop_flag.set()
            self.running = False
            
        signal.signal(signal.SIGINT, signal_handler)
        if hasattr(signal, 'SIGTERM'):
            signal.signal(signal.SIGTERM, signal_handler)
    
    def _create_midi_mapping(self) -> Dict[int, str]:
        """Create mapping from MIDI note numbers to keyboard keys."""
        mapping = {}
        for i, key in enumerate(self.KEYBOARD_LAYOUT):
            midi_note = self.base_midi_note + i
            mapping[midi_note] = key
        return mapping
    
    def _map_note_to_key(self, midi_note: int) -> Optional[str]:
        """Map MIDI note to keyboard key, transposing if out of range."""
        if midi_note in self.midi_to_key:
            return self.midi_to_key[midi_note]
        
        # Transpose to nearest available octave
        while midi_note < self.base_midi_note:
            midi_note += 12
        while midi_note >= self.base_midi_note + len(self.KEYBOARD_LAYOUT):
            midi_note -= 12
            
        return self.midi_to_key.get(midi_note)
    
    def _get_key_combination(self, key: str) -> Tuple[List, str]:
        """Get the actual key combination needed to produce the character."""
        symbol_map = {
            '!': '1', '@': '2', '#': '3', '$': '4', '%': '5',
            '^': '6', '&': '7', '*': '8', '(': '9', ')': '0'
        }
        
        if key in symbol_map:
            return ([Key.shift], symbol_map[key])
        elif key.isupper():
            return ([Key.shift], key.lower())
        else:
            return ([], key)
    
    def _group_notes_into_chords(self, notes: List[NoteEvent]) -> List[ChordGroup]:
        """Group notes by exact start time and assign to hands."""
        if not self.chord_aware:
            # Return individual notes as single-note "chords"
            return [ChordGroup(note.start_time, [], [note]) for note in notes]
        
        # Group by exact start time (0ms tolerance)
        time_groups = {}
        for note in notes:
            start_time = note.start_time
            if start_time not in time_groups:
                time_groups[start_time] = []
            time_groups[start_time].append(note)
        
        chord_groups = []
        for start_time, chord_notes in time_groups.items():
            # Split notes by hand based on MIDI note 60 (Middle C)
            left_hand = [n for n in chord_notes if n.midi_note <= self.HAND_SPLIT_NOTE]
            right_hand = [n for n in chord_notes if n.midi_note > self.HAND_SPLIT_NOTE]
            
            # Handle overflow (>5 notes per hand)
            if len(left_hand) > 5:
                # Move highest notes from left to right
                sorted_left = sorted(left_hand, key=lambda n: n.midi_note)
                overflow_count = len(left_hand) - 5
                overflow_notes = sorted_left[-overflow_count:]
                left_hand = sorted_left[:-overflow_count]
                right_hand.extend(overflow_notes)
            
            if len(right_hand) > 5:
                # Move lowest notes from right to left
                sorted_right = sorted(right_hand, key=lambda n: n.midi_note)
                overflow_count = len(right_hand) - 5
                overflow_notes = sorted_right[:overflow_count]
                right_hand = sorted_right[overflow_count:]
                left_hand.extend(overflow_notes)
            
            chord_groups.append(ChordGroup(start_time, left_hand, right_hand))
        
        return sorted(chord_groups, key=lambda c: c.start_time)
    
    def _apply_humanization_to_chord(self, chord: ChordGroup) -> List[Tuple[float, str, str, int, str]]:
        """Apply humanization to a chord group. Returns list of (time, event_type, key, midi_note, hand)."""
        events = []
        
        for hand, hand_notes in [('left', chord.left_hand_notes), ('right', chord.right_hand_notes)]:
            if not hand_notes:
                continue
                
            # Apply random order if enabled
            if self.random_timing:
                hand_notes = hand_notes.copy()
                random.shuffle(hand_notes)
            
            # Apply timing variations
            for i, note in enumerate(hand_notes):
                key = self._map_note_to_key(note.midi_note)
                if not key:
                    continue
                
                # Calculate timing variations
                press_delay = 0
                release_delay = 0
                
                if self.natural_timing or self.random_timing:
                    # Add small random delays to simulate human timing
                    if random.random() < 0.7:  # 70% chance of having some delay
                        press_delay = random.uniform(0, self.max_timing_variance)
                    
                    # Slight variation in note duration
                    if random.random() < 0.3:  # 30% chance of duration variation
                        duration_variance = random.uniform(-0.02, 0.02)
                        release_delay = duration_variance
                
                # For chords, add slight rolling effect if natural timing is enabled
                if self.natural_timing and len(hand_notes) > 1:
                    chord_roll_delay = i * 0.005  # 5ms between each note in chord
                    press_delay += chord_roll_delay
                
                # Ensure we don't go negative or extend too far
                actual_press_time = max(0, note.start_time + press_delay)
                actual_release_time = max(actual_press_time + 0.05, 
                                        note.end_time + release_delay)
                
                # Schedule press and release events
                events.append((actual_press_time, 'press', key, note.midi_note, hand))
                events.append((actual_release_time, 'release', key, note.midi_note, hand))
        
        return events
    
    def _press_key(self, key: str, midi_note: int):
        """Press a keyboard key with atomic shift operation to prevent bleed."""
        if self.stop_flag.is_set():
            return
            
        with self.lock:
            if key not in self.currently_pressed_keys:
                try:
                    modifiers, base_key = self._get_key_combination(key)
                    requires_shift = Key.shift in modifiers
                    
                    # Create separate controller instance for atomic operation
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
                except Exception:
                    pass
    
    def _release_key(self, key: str, midi_note: int):
        """Release a keyboard key atomically."""
        if self.stop_flag.is_set():
            return
            
        with self.lock:
            if key in self.currently_pressed_keys:
                try:
                    modifiers, base_key = self._get_key_combination(key)
                    
                    # Use main keyboard controller for release
                    self.keyboard.release(base_key)
                    time.sleep(0.001)  # Allow OS to process release
                    
                    self.currently_pressed_keys.discard(key)
                except Exception:
                    pass
    
    def _schedule_event(self, event_time: float, event_type: str, key: str, 
                       midi_note: int, hand: str = 'unknown'):
        """Schedule a keyboard event."""
        event = ScheduledEvent(event_time, event_type, key, midi_note, hand)
        heapq.heappush(self.event_queue, event)
    
    def _schedule_chord(self, chord: ChordGroup):
        """Schedule a chord to be played with humanization."""
        humanized_events = self._apply_humanization_to_chord(chord)
        
        for event_time, event_type, key, midi_note, hand in humanized_events:
            self._schedule_event(event_time, event_type, key, midi_note, hand)
    
    def _scheduler_loop(self):
        """Scheduler loop with timing compensation."""
        start_perf = time.perf_counter()
        accumulated_delay = 0.0
        
        while self.running and not self.stop_flag.is_set():
            current_time = time.perf_counter() - start_perf
            
            if self.event_queue and self.event_queue[0].time <= (current_time + accumulated_delay):
                event = heapq.heappop(self.event_queue)
                
                operation_start = time.perf_counter()
                
                if event.event_type == 'press':
                    self._press_key(event.key, event.midi_note)
                elif event.event_type == 'release':
                    self._release_key(event.key, event.midi_note)
                
                operation_time = time.perf_counter() - operation_start
                accumulated_delay += operation_time
            else:
                time.sleep(0.001)
    
    def _display_current_keys(self):
        """Display currently pressed keys with hand indicators."""
        while self.running and not self.stop_flag.is_set():
            with self.lock:
                current_keys = self.currently_pressed_keys.copy()
            
            if current_keys != self.last_displayed_keys:
                sys.stdout.write('\r' + ' ' * 100 + '\r')
                if current_keys:
                    key_display = ' '.join(sorted(current_keys))
                    prefix = "♪ Playing: " if not self.chord_aware else "🎹 Playing: "
                    sys.stdout.write(f'{prefix}{key_display}')
                else:
                    prefix = "♪" if not self.chord_aware else "🎹"
                    sys.stdout.write(prefix)
                sys.stdout.flush()
                self.last_displayed_keys = current_keys
            
            time.sleep(0.05)
    
    def load_midi_file(self, filename: str) -> List[ChordGroup]:
        """Load and parse MIDI file into chord groups."""
        try:
            mid = mido.MidiFile(filename)
        except Exception as e:
            raise ValueError(f"Error loading MIDI file: {e}")
        
        notes = []
        current_time = 0
        active_notes = {}
        
        for msg in mid:
            current_time += msg.time * self.tempo_scale
            
            if msg.type == 'note_on' and msg.velocity > 0:
                active_notes[msg.note] = {
                    'start_time': current_time,
                    'velocity': msg.velocity
                }
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
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
        
        # Group notes into chords
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
        
        # Force cleanup any stuck modifiers
        try:
            self.keyboard.release(Key.shift)
            self.keyboard.release(Key.ctrl)
            self.keyboard.release(Key.alt)
        except Exception:
            pass
        
        with self.lock:
            self.currently_pressed_keys.clear()
    
    def _cleanup_threads(self, timeout=2):
        """Clean up threads with aggressive termination."""
        print("\nStopping threads...")
        
        # Set stop flags first
        self.running = False
        self.stop_flag.set()
        
        # Give threads a moment to see the stop flags
        time.sleep(0.1)
        
        # Try to join scheduler thread
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=timeout)
            if self.scheduler_thread.is_alive():
                print("Warning: Scheduler thread did not stop cleanly")
        
        # Try to join display thread  
        if self.display_thread and self.display_thread.is_alive():
            self.display_thread.join(timeout=timeout)
            if self.display_thread.is_alive():
                print("Warning: Display thread did not stop cleanly")
        
        # Clear the display line
        sys.stdout.write('\r' + ' ' * 100 + '\r')
        sys.stdout.flush()
    
    def play_midi_file(self, filename: str, confirmation_enabled=True, countdown_enabled=True):
        """Play MIDI file with keyboard output."""
        chord_groups = self.load_midi_file(filename)
        
        if not chord_groups:
            print("No notes found!")
            return
        
        total_notes = sum(len(chord.all_notes) for chord in chord_groups)
        print(f"Loaded {total_notes} notes in {len(chord_groups)} chord groups")
        
        if self.chord_aware:
            left_hand_chords = sum(1 for chord in chord_groups if chord.left_hand_notes)
            right_hand_chords = sum(1 for chord in chord_groups if chord.right_hand_notes)
            print(f"Hand distribution: {left_hand_chords} left-hand, {right_hand_chords} right-hand chord groups")
        
        if confirmation_enabled and not self._get_user_confirmation():
            return
        
        if countdown_enabled:
            self.countdown()
            if self.stop_flag.is_set():
                return
        
        self.running = True
        self.start_time = time.perf_counter()
        
        # Start threads
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop)
        self.display_thread = threading.Thread(target=self._display_current_keys)
        
        self.scheduler_thread.start()
        self.display_thread.start()
        
        # Schedule all chord groups
        for chord in chord_groups:
            if self.stop_flag.is_set():
                break
            self._schedule_chord(chord)
        
        # Wait for playback completion
        if chord_groups and not self.stop_flag.is_set():
            total_duration = max(note.end_time for chord in chord_groups for note in chord.all_notes)
            end_time = time.perf_counter() + total_duration + 2
            
            while time.perf_counter() < end_time and not self.stop_flag.is_set():
                time.sleep(0.1)
        
        # Force cleanup regardless of how we got here
        self.running = False
        self.stop_flag.set()
        
        try:
            self._cleanup_threads()
            self._cleanup_keys()
        except Exception as e:
            print(f"Cleanup error: {e}")
        
        # Ensure we exit cleanly
        if self.stop_flag.is_set():
            print("Playback interrupted!")
        else:
            print("Done!")
        
        # Force exit if threads are still hanging
        sys.stdout.write('\n')
        sys.stdout.flush()

def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(description="MIDI-to-Keyboard Player with Humanization")
    
    parser.add_argument('midi_file', help='MIDI file to play')
    parser.add_argument('--tempo', type=float, default=1.0, help='Tempo scaling factor')
    parser.add_argument('-n', '--natural', action='store_true', 
                       help='Enable natural timing variations and chord rolling')
    parser.add_argument('-r', '--random-timing', action='store_true',
                       help='Enable random timing variations and note order')
    parser.add_argument('-c', '--chord-aware', action='store_true',
                       help='Enable chord grouping and hand assignment')
    parser.add_argument('--timing-variance', type=float, default=0.05,
                       help='Maximum timing variance in seconds (default: 0.05)')
    parser.add_argument('--no-confirmation', action='store_true', help='Skip confirmation')
    parser.add_argument('--no-countdown', action='store_true', help='Skip countdown')
    
    args = parser.parse_args()
    
    # Validate timing variance
    if args.timing_variance < 0 or args.timing_variance > 0.5:
        print("Error: Timing variance must be between 0.0 and 0.5 seconds")
        sys.exit(1)
    
    player = MIDIKeyboardPlayer(
        tempo_scale=args.tempo,
        natural_timing=args.natural,
        random_timing=args.random_timing,
        chord_aware=args.chord_aware,
        max_timing_variance=args.timing_variance
    )
    
    # Show configuration
    features = []
    if args.natural:
        features.append("natural timing")
    if args.random_timing:
        features.append("random timing")
    if args.chord_aware:
        features.append("chord-aware hand assignment")
    
    if features:
        print(f"Enabled features: {', '.join(features)}")
    
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
        # Ensure clean exit
        try:
            player.stop_flag.set()
            player.running = False
            time.sleep(0.1)  # Give threads time to stop
        except:
            pass
        
        # Force exit if still hanging
        sys.exit(0)

if __name__ == "__main__":
    main()
