#!/usr/bin/env python3
"""
MIDI-to-Keyboard Player - Clean Version
Maps MIDI notes to computer keyboard keys.
"""

import argparse
import mido
import time
import threading
import sys
import heapq
import signal
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
class ScheduledEvent:
    """Represents a scheduled keyboard event."""
    time: float
    event_type: str  # 'press' or 'release'
    key: str
    midi_note: int
    
    def __lt__(self, other):
        return self.time < other.time

class MIDIKeyboardPlayer:
    """Main MIDI keyboard player class."""
    
    KEYBOARD_LAYOUT = "1!2@34$5%6^78*9(0qQwWeErtTyYuiIoOpPasSdDfgGhHjJklLzZxcCvVbBnm"
    C4_INDEX = 24
    C4_MIDI = 60
    
    def __init__(self, tempo_scale=1.0, mistake_chance=0.0):
        self.keyboard = Controller()
        self.tempo_scale = tempo_scale
        self.mistake_chance = mistake_chance
        
        # MIDI to keyboard mapping
        self.base_midi_note = self.C4_MIDI - self.C4_INDEX
        self.midi_to_key = self._create_midi_mapping()
        
        # White key positions for mistake generation
        self.white_keys = set(['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', 
                              'q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p',
                              'a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l',
                              'z', 'x', 'c', 'v', 'b', 'n', 'm'])
        
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
    
    def _get_adjacent_white_key(self, key: str) -> Optional[str]:
        """Get an adjacent white key for mistake generation."""
        if key not in self.white_keys:
            return None
            
        # Find current key position in layout
        try:
            current_pos = self.KEYBOARD_LAYOUT.index(key)
        except ValueError:
            return None
        
        # Look for adjacent white keys
        for offset in [-1, 1]:  # Left and right
            new_pos = current_pos + offset
            if 0 <= new_pos < len(self.KEYBOARD_LAYOUT):
                adjacent_key = self.KEYBOARD_LAYOUT[new_pos]
                if adjacent_key in self.white_keys:
                    return adjacent_key
        return None
    
    def _should_make_mistake(self, note_duration: float) -> bool:
        """Determine if a mistake should occur based on chance and note duration."""
        if self.mistake_chance <= 0:
            return False
        
        import random
        # Higher chance for shorter notes (more prone to mistakes)
        duration_factor = 1.5 if note_duration < 0.25 else 1.0  # 16th notes at 120bpm ≈ 0.125s
        actual_chance = self.mistake_chance * duration_factor
        
        return random.random() < actual_chance
    
    def _generate_mistake_events(self, note_event: NoteEvent, correct_key: str) -> List[Tuple[float, str, str, int]]:
        """Generate mistake events for a note. Returns list of (time, event_type, key, midi_note)."""
        import random
        
        events = []
        duration = note_event.duration
        start_time = note_event.start_time
        end_time = note_event.end_time
        
        # Only make mistakes on white keys (simpler to handle)
        if correct_key not in self.white_keys:
            # Just play the correct note
            events.append((start_time, 'press', correct_key, note_event.midi_note))
            events.append((end_time, 'release', correct_key, note_event.midi_note))
        
        return events
    
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
        
        mistake_type = random.choice(['wrong_key', 'adjacent_double', 'wrong_quick_correct'])
        
        if mistake_type == 'wrong_key' and duration < 0.3:
            # Short note: just hit wrong key instead
            adjacent_key = self._get_adjacent_white_key(correct_key)
            if adjacent_key:
                events.append((start_time, 'press', adjacent_key, note_event.midi_note))
                events.append((end_time, 'release', adjacent_key, note_event.midi_note))
            else:
                # Fallback to correct key
                events.append((start_time, 'press', correct_key, note_event.midi_note))
                events.append((end_time, 'release', correct_key, note_event.midi_note))
                
        elif mistake_type == 'adjacent_double':
            # Press both adjacent keys (common mistake)
            adjacent_key = self._get_adjacent_white_key(correct_key)
            if adjacent_key:
                # Press both keys at start
                events.append((start_time, 'press', correct_key, note_event.midi_note))
                events.append((start_time + 0.01, 'press', adjacent_key, note_event.midi_note))
                
                if duration > 0.5:  # Long note: release wrong key early
                    events.append((start_time + 0.05, 'release', adjacent_key, note_event.midi_note))
                    events.append((end_time, 'release', correct_key, note_event.midi_note))
                else:  # Short note: keep both pressed
                    events.append((end_time, 'release', correct_key, note_event.midi_note))
                    events.append((end_time + 0.01, 'release', adjacent_key, note_event.midi_note))
            else:
                # Fallback to correct key
                events.append((start_time, 'press', correct_key, note_event.midi_note))
                events.append((end_time, 'release', correct_key, note_event.midi_note))
                
        elif mistake_type == 'wrong_quick_correct':
            # Press wrong key, quickly correct to right key
            adjacent_key = self._get_adjacent_white_key(correct_key)
            if adjacent_key and duration > 0.1:
                mistake_duration = min(0.05, duration * 0.3)  # Quick mistake
                
                # Press wrong key first
                events.append((start_time, 'press', adjacent_key, note_event.midi_note))
                events.append((start_time + mistake_duration, 'release', adjacent_key, note_event.midi_note))
                
                # Then press correct key
                events.append((start_time + mistake_duration, 'press', correct_key, note_event.midi_note))
                events.append((end_time, 'release', correct_key, note_event.midi_note))
            else:
                # Fallback to correct key
                events.append((start_time, 'press', correct_key, note_event.midi_note))
                events.append((end_time, 'release', correct_key, note_event.midi_note))
        
        return events
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
    
    def _schedule_event(self, event_time: float, event_type: str, key: str, midi_note: int):
        """Schedule a keyboard event."""
        event = ScheduledEvent(event_time, event_type, key, midi_note)
        heapq.heappush(self.event_queue, event)
    
    def _schedule_note(self, note_event: NoteEvent):
        """Schedule a note to be played, with optional mistakes."""
        key = self._map_note_to_key(note_event.midi_note)
        if not key:
            return
        
        # Check if we should make a mistake
        if self._should_make_mistake(note_event.duration):
            # Generate mistake events
            events = self._generate_mistake_events(note_event, key)
            for event_time, event_type, event_key, midi_note in events:
                self._schedule_event(event_time, event_type, event_key, midi_note)
        else:
            # Normal note playing
            self._schedule_event(note_event.start_time, 'press', key, note_event.midi_note)
            self._schedule_event(note_event.end_time, 'release', key, note_event.midi_note)
    
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
        """Display currently pressed keys."""
        while self.running and not self.stop_flag.is_set():
            with self.lock:
                current_keys = self.currently_pressed_keys.copy()
            
            if current_keys != self.last_displayed_keys:
                sys.stdout.write('\r' + ' ' * 80 + '\r')
                if current_keys:
                    key_display = ' '.join(sorted(current_keys))
                    sys.stdout.write(f'Playing: {key_display}')
                else:
                    sys.stdout.write('♪')
                sys.stdout.flush()
                self.last_displayed_keys = current_keys
            
            time.sleep(0.05)
    
    def load_midi_file(self, filename: str) -> List[NoteEvent]:
        """Load and parse MIDI file."""
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
        
        return sorted(notes, key=lambda n: n.start_time)
    
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
        sys.stdout.write('\r' + ' ' * 80 + '\r')
        sys.stdout.flush()
    
    def play_midi_file(self, filename: str, confirmation_enabled=True, countdown_enabled=True):
        """Play MIDI file with keyboard output."""
        notes = self.load_midi_file(filename)
        
        if not notes:
            print("No notes found!")
            return
        
        print(f"Loaded {len(notes)} notes")
        
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
        
        # Schedule all notes
        for note in notes:
            if self.stop_flag.is_set():
                break
            self._schedule_note(note)
        
        # Wait for playback completion
        if notes and not self.stop_flag.is_set():
            total_duration = max(note.end_time for note in notes)
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
    parser = argparse.ArgumentParser(description="MIDI-to-Keyboard Player")
    
    parser.add_argument('midi_file', help='MIDI file to play')
    parser.add_argument('--tempo', type=float, default=1.0, help='Tempo scaling factor')
    parser.add_argument('--mistakes', type=float, default=0.0, help='Mistake chance (0.0-1.0, e.g. 0.05 = 5%%)')
    parser.add_argument('--no-confirmation', action='store_true', help='Skip confirmation')
    parser.add_argument('--no-countdown', action='store_true', help='Skip countdown')
    
    args = parser.parse_args()
    
    # Validate mistake chance
    if args.mistakes < 0 or args.mistakes > 1:
        print("Error: Mistake chance must be between 0.0 and 1.0")
        sys.exit(1)
    
    player = MIDIKeyboardPlayer(tempo_scale=args.tempo, mistake_chance=args.mistakes)
    
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