#!/usr/bin/env python3
"""
MIDI to Piano-Style Keyboard Player - Enhanced Human-like Version

Reads a MIDI file and simulates pressing computer keyboard keys
in a piano-like layout with 5 octave range (2 octaves up and down from center).

Required libraries:
    pip install mido pynput

Usage:
    python midi_keyboard_player.py your_file.mid [options]
"""

import mido
import time as time_module
import sys
import os
import threading
import queue
import random
import argparse

try:
    import ctypes
    from ctypes import wintypes
    WINDOWS_AVAILABLE = True
except ImportError:
    WINDOWS_AVAILABLE = False

try:
    from pynput.keyboard import Key, Controller
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False


class PynputKeyboardController:
    def __init__(self):
        if not PYNPUT_AVAILABLE:
            raise ImportError("pynput library not available")
        self.controller = Controller()
    
    def send_key(self, key_char, duration=0.008):
        try:
            if key_char == '!':
                with self.controller.pressed(Key.shift):
                    self.controller.press('1')
                    time_module.sleep(duration)
                    self.controller.release('1')
            elif key_char == '@':
                with self.controller.pressed(Key.shift):
                    self.controller.press('2')
                    time_module.sleep(duration)
                    self.controller.release('2')
            elif key_char == '$':
                with self.controller.pressed(Key.shift):
                    self.controller.press('4')
                    time_module.sleep(duration)
                    self.controller.release('4')
            elif key_char == '%':
                with self.controller.pressed(Key.shift):
                    self.controller.press('5')
                    time_module.sleep(duration)
                    self.controller.release('5')
            elif key_char == '^':
                with self.controller.pressed(Key.shift):
                    self.controller.press('6')
                    time_module.sleep(duration)
                    self.controller.release('6')
            elif key_char == '*':
                with self.controller.pressed(Key.shift):
                    self.controller.press('8')
                    time_module.sleep(duration)
                    self.controller.release('8')
            elif key_char == '(':
                with self.controller.pressed(Key.shift):
                    self.controller.press('9')
                    time_module.sleep(duration)
                    self.controller.release('9')
            elif key_char.isupper():
                with self.controller.pressed(Key.shift):
                    self.controller.press(key_char.lower())
                    time_module.sleep(duration)
                    self.controller.release(key_char.lower())
            else:
                self.controller.press(key_char)
                time_module.sleep(duration)
                self.controller.release(key_char)
            return True
        except Exception as e:
            print(f"Error sending key '{key_char}': {e}")
            return False

    def send_simultaneous_keys(self, key_list, duration=0.008):
        try:
            pressed_keys = []
            shift_needed = False
            
            for key_char in key_list:
                if key_char in ['!', '@', '$', '%', '^', '*', '('] or key_char.isupper():
                    if not shift_needed:
                        self.controller.press(Key.shift)
                        shift_needed = True
                    
                    if key_char == '!':
                        self.controller.press('1')
                        pressed_keys.append('1')
                    elif key_char == '@':
                        self.controller.press('2')
                        pressed_keys.append('2')
                    elif key_char == '$':
                        self.controller.press('4')
                        pressed_keys.append('4')
                    elif key_char == '%':
                        self.controller.press('5')
                        pressed_keys.append('5')
                    elif key_char == '^':
                        self.controller.press('6')
                        pressed_keys.append('6')
                    elif key_char == '*':
                        self.controller.press('8')
                        pressed_keys.append('8')
                    elif key_char == '(':
                        self.controller.press('9')
                        pressed_keys.append('9')
                    elif key_char.isupper():
                        self.controller.press(key_char.lower())
                        pressed_keys.append(key_char.lower())
                else:
                    self.controller.press(key_char)
                    pressed_keys.append(key_char)
            
            time_module.sleep(duration)
            
            for key in reversed(pressed_keys):
                self.controller.release(key)
            
            if shift_needed:
                self.controller.release(Key.shift)
            
            return True
        except Exception as e:
            print(f"Error sending simultaneous keys {key_list}: {e}")
            return False


class WindowsKeyboardController:
    def __init__(self):
        if not WINDOWS_AVAILABLE:
            raise ImportError("Windows ctypes not available")
        
        self.INPUT_KEYBOARD = 1
        self.KEYEVENTF_KEYUP = 0x0002
        self.VK_SHIFT = 0x10
        
        self.KEY_MAP = {
            '1': 0x31, '2': 0x32, '3': 0x33, '4': 0x34, '5': 0x35,
            '6': 0x36, '7': 0x37, '8': 0x38, '9': 0x39, '0': 0x30,
            'q': 0x51, 'w': 0x57, 'e': 0x45, 'r': 0x52, 't': 0x54,
            'y': 0x59, 'u': 0x55, 'i': 0x49, 'o': 0x4F, 'p': 0x50,
            'a': 0x41, 's': 0x53, 'd': 0x44, 'f': 0x46, 'g': 0x47,
            'h': 0x48, 'j': 0x4A, 'k': 0x4B, 'l': 0x4C,
            'z': 0x5A, 'x': 0x58, 'c': 0x43, 'v': 0x56, 'b': 0x42,
            'n': 0x4E, 'm': 0x4D
        }
        
        self._init_structures()
    
    def _init_structures(self):
        class KEYBDINPUT(ctypes.Structure):
            _fields_ = [
                ("wVk", wintypes.WORD),
                ("wScan", wintypes.WORD),
                ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("dwExtraInfo", ctypes.POINTER(wintypes.ULONG))
            ]
        
        class MOUSEINPUT(ctypes.Structure):
            _fields_ = [
                ("dx", wintypes.LONG),
                ("dy", wintypes.LONG),
                ("mouseData", wintypes.DWORD),
                ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("dwExtraInfo", ctypes.POINTER(wintypes.ULONG))
            ]
        
        class HARDWAREINPUT(ctypes.Structure):
            _fields_ = [
                ("uMsg", wintypes.DWORD),
                ("wParamL", wintypes.WORD),
                ("wParamH", wintypes.WORD)
            ]
        
        class INPUT_UNION(ctypes.Union):
            _fields_ = [
                ("ki", KEYBDINPUT),
                ("mi", MOUSEINPUT),
                ("hi", HARDWAREINPUT)
            ]
        
        class INPUT(ctypes.Structure):
            _fields_ = [
                ("type", wintypes.DWORD),
                ("union", INPUT_UNION)
            ]
        
        self.KEYBDINPUT = KEYBDINPUT
        self.INPUT_UNION = INPUT_UNION
        self.INPUT = INPUT
        self.user32 = ctypes.windll.user32
    
    def _send_key_event(self, vk_code, key_up=False):
        try:
            extra = ctypes.pointer(wintypes.ULONG(0))
            ki = self.KEYBDINPUT(
                wVk=vk_code,
                wScan=0,
                dwFlags=self.KEYEVENTF_KEYUP if key_up else 0,
                time=0,
                dwExtraInfo=extra
            )
            
            union = self.INPUT_UNION()
            union.ki = ki
            
            input_struct = self.INPUT(
                type=self.INPUT_KEYBOARD,
                union=union
            )
            
            result = self.user32.SendInput(1, ctypes.byref(input_struct), ctypes.sizeof(self.INPUT))
            return result == 1
        except Exception:
            return False
    
    def send_key(self, key_char, duration=0.008):
        try:
            symbol_map = {
                '!': '1', '@': '2', '$': '4', '%': '5', 
                '^': '6', '*': '8', '(': '9'
            }
            
            if key_char in symbol_map:
                base_key = symbol_map[key_char]
                if base_key in self.KEY_MAP:
                    vk_code = self.KEY_MAP[base_key]
                    success = True
                    success &= self._send_key_event(self.VK_SHIFT, False)
                    success &= self._send_key_event(vk_code, False)
                    time_module.sleep(duration)
                    success &= self._send_key_event(vk_code, True)
                    success &= self._send_key_event(self.VK_SHIFT, True)
                    return success
                return False
            
            key_lower = key_char.lower()
            if key_lower not in self.KEY_MAP:
                return False
            
            vk_code = self.KEY_MAP[key_lower]
            success = True
            
            if key_char.isupper():
                success &= self._send_key_event(self.VK_SHIFT, False)
            
            success &= self._send_key_event(vk_code, False)
            time_module.sleep(duration)
            success &= self._send_key_event(vk_code, True)
            
            if key_char.isupper():
                success &= self._send_key_event(self.VK_SHIFT, True)
            
            return success
        except Exception:
            return False

    def send_simultaneous_keys(self, key_list, duration=0.008):
        try:
            symbol_map = {
                '!': '1', '@': '2', '$': '4', '%': '5', 
                '^': '6', '*': '8', '(': '9'
            }
            
            keys_to_press = []
            shift_needed = False
            
            for key_char in key_list:
                if key_char in symbol_map:
                    keys_to_press.append(self.KEY_MAP[symbol_map[key_char]])
                    shift_needed = True
                elif key_char.isupper():
                    keys_to_press.append(self.KEY_MAP[key_char.lower()])
                    shift_needed = True
                elif key_char.lower() in self.KEY_MAP:
                    keys_to_press.append(self.KEY_MAP[key_char.lower()])
            
            success = True
            
            if shift_needed:
                success &= self._send_key_event(self.VK_SHIFT, False)
            
            for vk_code in keys_to_press:
                success &= self._send_key_event(vk_code, False)
            
            time_module.sleep(duration)
            
            for vk_code in reversed(keys_to_press):
                success &= self._send_key_event(vk_code, True)
            
            if shift_needed:
                success &= self._send_key_event(self.VK_SHIFT, True)
            
            return success
        except Exception:
            return False


class PianoKeyboardPlayer:
    def __init__(self, center_note=60, natural_timing=False, adaptive_timing=True, delay_scale=1.0, mistake_rate=0.0, simultaneous_chords=False, randomize_chord_order=True, random_simultaneous_chords=False):
        self.center_note = center_note
        self.natural_timing = natural_timing
        self.adaptive_timing = adaptive_timing
        self.delay_scale = delay_scale
        self.mistake_rate = mistake_rate
        self.simultaneous_chords = simultaneous_chords
        self.randomize_chord_order = randomize_chord_order
        self.random_simultaneous_chords = random_simultaneous_chords
        self.is_playing = False
        self.should_stop = False
        
        self.keyboard_controller = None
        self._init_keyboard_controller()
        
        self.thinking_pause_chance = 0.015
        self.last_thinking_pause = 0
        
        self.white_keys = list("1234567890qwertyuiopasdfghjklzxcvbnm")
        self.black_keys = list("!@$%^*(QWETYIOPSDGHJLZCVB")
        
        self.black_key_map = {
            '1': '!', '2': '@', '4': '$', '5': '%', '6': '^', '8': '*', '9': '(',
            'q': 'Q', 'w': 'W', 'e': 'E', 't': 'T', 'y': 'Y',
            'i': 'I', 'o': 'O', 'p': 'P',
            's': 'S', 'd': 'D', 'g': 'G', 'h': 'H',
            'l': 'L', 'z': 'Z', 'c': 'C', 'v': 'V', 'b': 'B'
        }
        
        self.key_queue = queue.Queue()
        self.key_thread = None
        
        self.note_to_key_map = {}
        self.key_to_note_map = {}
        self._build_piano_mapping()
        self._build_adjacent_key_maps()
    
    def _init_keyboard_controller(self):
        controllers_tried = []
        
        if PYNPUT_AVAILABLE:
            try:
                self.keyboard_controller = PynputKeyboardController()
                return
            except Exception as e:
                controllers_tried.append(f"pynput: {e}")
        
        if WINDOWS_AVAILABLE:
            try:
                self.keyboard_controller = WindowsKeyboardController()
                return
            except Exception as e:
                controllers_tried.append(f"Windows SendInput: {e}")
        
        print("ERROR: No keyboard controller available!")
        for attempt in controllers_tried:
            print(f"  Failed: {attempt}")
        print("\nTo fix: pip install pynput")
        raise RuntimeError("No keyboard controller available")
    
    def _build_adjacent_key_maps(self):
        white_key_sequence = "1234567890qwertyuiopasdfghjklzxcvbnm"
        
        self.adjacent_white_keys = {}
        for i, key in enumerate(white_key_sequence):
            adjacent = []
            if i > 0:
                adjacent.append(white_key_sequence[i-1])
            if i < len(white_key_sequence) - 1:
                adjacent.append(white_key_sequence[i+1])
            self.adjacent_white_keys[key] = adjacent
    
    def _build_piano_mapping(self):
        start_note = self.center_note - 24
        white_key_index = 0
        current_note = start_note
        
        while white_key_index < len(self.white_keys) and current_note <= 127:
            note_in_octave = current_note % 12
            
            if note_in_octave in [0, 2, 4, 5, 7, 9, 11]:
                if white_key_index < len(self.white_keys):
                    white_key = self.white_keys[white_key_index]
                    self.note_to_key_map[current_note] = white_key
                    self.key_to_note_map[white_key] = current_note
                    
                    if note_in_octave in [0, 2, 5, 7, 9]:
                        sharp_note = current_note + 1
                        if sharp_note <= 127 and white_key in self.black_key_map:
                            black_key = self.black_key_map[white_key]
                            self.note_to_key_map[sharp_note] = black_key
                            self.key_to_note_map[black_key] = sharp_note
                    
                    white_key_index += 1
            
            current_note += 1
    
    def find_nearest_playable_note(self, target_note):
        if target_note in self.note_to_key_map:
            return target_note
        
        available_notes = list(self.note_to_key_map.keys())
        if not available_notes:
            return None
        
        target_note_in_octave = target_note % 12
        same_note_candidates = [note for note in available_notes if note % 12 == target_note_in_octave]
        
        if same_note_candidates:
            return min(same_note_candidates, key=lambda x: abs(x - target_note))
        
        return min(available_notes, key=lambda x: abs(x - target_note))
    
    def get_note_name(self, midi_note):
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        octave = (midi_note // 12) - 1
        note = note_names[midi_note % 12]
        return f"{note}{octave}"
    
    def apply_mistakes(self, keys):
        if not self.natural_timing or self.mistake_rate == 0.0:
            return keys
        
        if isinstance(keys, str):
            if random.random() < self.mistake_rate:
                return self._make_single_key_mistake(keys)
            return keys
        else:
            result_keys = []
            for key in keys:
                if random.random() < self.mistake_rate:
                    mistake_result = self._make_single_key_mistake(key)
                    if mistake_result is None:
                        continue
                    elif isinstance(mistake_result, list):
                        result_keys.extend(mistake_result)
                    else:
                        result_keys.append(mistake_result)
                else:
                    result_keys.append(key)
            
            return result_keys if result_keys else keys
    
    def _make_single_key_mistake(self, key):
        if key in self.black_keys and random.random() > 0.1:
            return key
        
        if key in self.adjacent_white_keys:
            mistake_type = random.choice([
                "adjacent", "adjacent", "add_adjacent", "skip", "random_white"
            ])
            
            if mistake_type == "adjacent" and self.adjacent_white_keys[key]:
                return random.choice(self.adjacent_white_keys[key])
            elif mistake_type == "add_adjacent" and self.adjacent_white_keys[key]:
                return [key, random.choice(self.adjacent_white_keys[key])]
            elif mistake_type == "skip":
                return None
            elif mistake_type == "random_white":
                white_keys = self.white_keys
                if key in white_keys:
                    current_idx = white_keys.index(key)
                    start_idx = max(0, current_idx - 3)
                    end_idx = min(len(white_keys), current_idx + 4)
                    nearby_keys = white_keys[start_idx:end_idx]
                    if len(nearby_keys) > 1:
                        nearby_keys.remove(key)
                        return random.choice(nearby_keys)
        
        return key
    
    def should_play_chord_simultaneously(self, chord_size, avg_velocity=64):
        if not self.random_simultaneous_chords:
            return False
        
        if chord_size <= 2:
            base_prob = 0.15
        elif chord_size == 3:
            base_prob = 0.35
        elif chord_size <= 6:
            base_prob = 0.65
        else:
            base_prob = 0.80
        
        if avg_velocity > 100:
            base_prob += 0.20
        elif avg_velocity > 80:
            base_prob += 0.10
        elif avg_velocity < 40:
            base_prob -= 0.15
        
        if self.natural_timing:
            variation = random.uniform(-0.15, 0.15)
            base_prob += variation
        
        base_prob = max(0.0, min(1.0, base_prob))
        return random.random() < base_prob
    
    def randomize_chord_keys(self, keys, velocities=None):
        if not self.randomize_chord_order or len(keys) <= 1:
            return keys, velocities
        
        if velocities:
            key_vel_pairs = list(zip(keys, velocities))
        else:
            key_vel_pairs = [(key, None) for key in keys]
        
        if len(keys) <= 3:
            patterns = ['bottom_up', 'top_down', 'outside_in', 'random']
            pattern = random.choice(patterns)
            
            if pattern == 'bottom_up':
                key_vel_pairs.sort(key=lambda x: self.white_keys.index(x[0]) if x[0] in self.white_keys else len(self.white_keys))
            elif pattern == 'top_down':
                key_vel_pairs.sort(key=lambda x: self.white_keys.index(x[0]) if x[0] in self.white_keys else len(self.white_keys), reverse=True)
            elif pattern == 'outside_in':
                if len(keys) >= 3:
                    sorted_pairs = sorted(key_vel_pairs, key=lambda x: self.white_keys.index(x[0]) if x[0] in self.white_keys else len(self.white_keys))
                    reordered = []
                    left = 0
                    right = len(sorted_pairs) - 1
                    while left <= right:
                        if left == right:
                            reordered.append(sorted_pairs[left])
                        else:
                            reordered.append(sorted_pairs[left])
                            if right != left:
                                reordered.append(sorted_pairs[right])
                        left += 1
                        right -= 1
                    key_vel_pairs = reordered
            else:
                random.shuffle(key_vel_pairs)
        else:
            patterns = ['random', 'partial_bottom_up', 'split_hands', 'clustered']
            pattern = random.choice(patterns)
            
            if pattern == 'random':
                random.shuffle(key_vel_pairs)
            elif pattern == 'partial_bottom_up':
                key_vel_pairs.sort(key=lambda x: self.white_keys.index(x[0]) if x[0] in self.white_keys else len(self.white_keys))
                for i in range(len(key_vel_pairs) - 1):
                    if random.random() < 0.3:
                        key_vel_pairs[i], key_vel_pairs[i + 1] = key_vel_pairs[i + 1], key_vel_pairs[i]
            elif pattern == 'split_hands':
                sorted_pairs = sorted(key_vel_pairs, key=lambda x: self.white_keys.index(x[0]) if x[0] in self.white_keys else len(self.white_keys))
                mid_point = len(sorted_pairs) // 2
                left_hand = sorted_pairs[:mid_point]
                right_hand = sorted_pairs[mid_point:]
                result = []
                max_len = max(len(left_hand), len(right_hand))
                for i in range(max_len):
                    if i < len(left_hand):
                        result.append(left_hand[i])
                    if i < len(right_hand):
                        result.append(right_hand[i])
                key_vel_pairs = result
            else:
                random.shuffle(key_vel_pairs)
        
        if velocities:
            new_keys, new_velocities = zip(*key_vel_pairs)
            return list(new_keys), list(new_velocities)
        else:
            new_keys = [pair[0] for pair in key_vel_pairs]
            return new_keys, None
    
    def send_single_key(self, key_char, duration=0.008, velocity=64):
        try:
            if self.natural_timing:
                micro_delay = random.uniform(0.0, 0.03) * self.delay_scale
                time_module.sleep(micro_delay)
                
                velocity_delay = (velocity / 127.0) * 0.02 * self.delay_scale
                time_module.sleep(velocity_delay)
                
                if (random.random() < self.thinking_pause_chance and 
                    self.last_thinking_pause > 5):
                    thinking_delay = random.uniform(0.05, 0.1) * self.delay_scale
                    time_module.sleep(thinking_delay)
                    self.last_thinking_pause = 0
                else:
                    self.last_thinking_pause += 1
                
                duration = duration + random.uniform(-0.002, 0.005)
                duration = max(0.003, duration)
            
            self.keyboard_controller.send_key(key_char, duration)
        except Exception as e:
            print(f"Error sending key {key_char}: {e}")
    
    def send_batch_keys(self, key_list, duration=0.006, velocities=None):
        chord_size = len(key_list)
        randomized_keys = key_list
        randomized_velocities = velocities
        
        try:
            randomized_keys, randomized_velocities = self.randomize_chord_keys(key_list, velocities)
            
            avg_velocity = sum(randomized_velocities) // len(randomized_velocities) if randomized_velocities else 64
            play_simultaneously = False
            
            if self.simultaneous_chords:
                play_simultaneously = True
            elif self.random_simultaneous_chords:
                play_simultaneously = self.should_play_chord_simultaneously(chord_size, avg_velocity)
            
            if play_simultaneously and hasattr(self.keyboard_controller, 'send_simultaneous_keys'):
                if self.natural_timing:
                    micro_delay = random.uniform(0.0, 0.02) * self.delay_scale
                    time_module.sleep(micro_delay)
                    
                    if chord_size > 6:
                        duration = duration * 1.5
                
                self.keyboard_controller.send_simultaneous_keys(randomized_keys, duration)
            else:
                if self.natural_timing:
                    micro_delay = random.uniform(0.0, 0.02) * self.delay_scale
                    time_module.sleep(micro_delay)
                    
                    if chord_size <= 3:
                        chord_spread = random.uniform(0.005, 0.015) * self.delay_scale
                    elif chord_size <= 6:
                        chord_spread = random.uniform(0.010, 0.025) * self.delay_scale
                    else:
                        chord_spread = random.uniform(0.015, 0.035) * self.delay_scale
                        
                        if random.random() < 0.3:
                            complex_pause = random.uniform(0.05, 0.12) * self.delay_scale
                            time_module.sleep(complex_pause)
                else:
                    if chord_size <= 3:
                        chord_spread = 0.001
                    elif chord_size <= 6:
                        chord_spread = 0.002
                    else:
                        chord_spread = 0.003
                
                for i, key_char in enumerate(randomized_keys):
                    self.keyboard_controller.send_key(key_char, duration)
                    
                    if i < len(randomized_keys) - 1:
                        time_module.sleep(chord_spread)
        except Exception as e:
            print(f"Error sending batch keys {randomized_keys}: {e}")
    
    def key_worker(self):
        while self.is_playing:
            try:
                key_event = self.key_queue.get(timeout=0.1)
                
                if key_event is None:
                    break
                
                event_type, keys, duration, velocity_data = key_event
                
                if event_type == "single":
                    velocity = velocity_data if velocity_data else 64
                    self.send_single_key(keys, duration, velocity)
                elif event_type == "batch":
                    velocities = velocity_data if velocity_data else None
                    self.send_batch_keys(keys, duration, velocities)
                
                self.key_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in key worker: {e}")
    
    def queue_key_event(self, keys, duration=0.008, velocity_data=None):
        keys_with_mistakes = self.apply_mistakes(keys)
        
        if keys_with_mistakes is None:
            return
        
        if isinstance(keys_with_mistakes, list) and len(keys_with_mistakes) > 1:
            self.key_queue.put(("batch", keys_with_mistakes, duration, velocity_data))
        else:
            single_key = keys_with_mistakes[0] if isinstance(keys_with_mistakes, list) else keys_with_mistakes
            velocity = velocity_data[0] if isinstance(velocity_data, list) else velocity_data
            self.key_queue.put(("single", single_key, duration, velocity))
    
    def load_midi_file(self, filename):
        try:
            midi_file = mido.MidiFile(filename)
            print(f"MIDI File        : {os.path.basename(filename)} ({midi_file.length:.1f}s, {len(midi_file.tracks)} tracks)")
            return midi_file
        except Exception as e:
            print(f"Error loading MIDI file: {e}")
            return None
    
    def show_piano_layout(self):
        print(f"\n{'='*60}")
        print("MIDI PIANO KEYBOARD PLAYER - ENHANCED HUMAN-LIKE VERSION")
        print("="*60)
        
        print(f"Center Note      : {self.get_note_name(self.center_note)} (MIDI {self.center_note})")
        if self.note_to_key_map:
            min_note = min(self.note_to_key_map.keys())
            max_note = max(self.note_to_key_map.keys())
            print(f"Range            : {self.get_note_name(min_note)} to {self.get_note_name(max_note)}")
        
        controller_type = type(self.keyboard_controller).__name__
        if "Windows" in controller_type:
            print(f"Keyboard Control : Windows SendInput API")
        elif "Pynput" in controller_type:
            print(f"Keyboard Control : pynput library")
        
        timing_features = []
        if self.natural_timing:
            timing_features.append("Natural variations")
            if self.delay_scale != 1.0:
                timing_features.append(f"Delay scale {self.delay_scale}x")
            if self.mistake_rate > 0.0:
                timing_features.append(f"Mistake rate {self.mistake_rate:.0%}")
        if self.adaptive_timing:
            timing_features.append("Adaptive chords")
        if self.simultaneous_chords:
            timing_features.append("All chords simultaneous")
        elif self.random_simultaneous_chords:
            timing_features.append("Random simultaneous chords")
        if self.randomize_chord_order:
            timing_features.append("Randomized chord order")
        
        if timing_features:
            print(f"Timing Features  : {' + '.join(timing_features)}")
        else:
            print(f"Timing Features  : Precise mechanical")
        
        print(f"Keyboard Layout  : {len(self.white_keys)} white + {len(self.black_keys)} black keys")
        print("="*60)
    
    def analyze_midi_file(self, midi_file):
        notes_found = set()
        total_notes = 0
        
        for track_idx, track in enumerate(midi_file.tracks):
            for msg in track:
                if msg.type == 'note_on' and msg.velocity > 0:
                    notes_found.add(msg.note)
                    total_notes += 1
        
        direct_mapped = sum(1 for note in notes_found if note in self.note_to_key_map)
        octave_mapped = len(notes_found) - direct_mapped
        total_playable = direct_mapped + octave_mapped
        
        if notes_found:
            direct_percent = (direct_mapped / len(notes_found)) * 100
            total_percent = (total_playable / len(notes_found)) * 100
            
            print(f"MIDI Analysis    : {len(midi_file.tracks)} tracks, {total_notes} notes, {len(notes_found)} unique")
            print(f"Playability      : {direct_mapped} direct ({direct_percent:.0f}%) + {octave_mapped} mapped = {total_percent:.0f}% playable")
    
    def play_midi_file(self, midi_file, speed_multiplier=1.0):
        print("="*60)
        print("PLAYBACK STARTED - Press Ctrl+C to stop")
        print("="*60)
        
        self.is_playing = True
        self.should_stop = False
        
        self.key_thread = threading.Thread(target=self.key_worker, daemon=True)
        self.key_thread.start()
        
        try:
            note_events = []
            tempo_events = []
            
            for track_idx, track in enumerate(midi_file.tracks):
                current_tick = 0
                for msg in track:
                    current_tick += msg.time
                    if msg.type == 'note_on' and msg.velocity > 0:
                        note_events.append((current_tick, msg.note, msg.velocity, track_idx))
                    elif msg.type == 'set_tempo':
                        tempo_events.append((current_tick, msg.tempo))
            
            note_events.sort(key=lambda x: x[0])
            tempo_events.sort(key=lambda x: x[0])
            
            if not note_events:
                print("No notes found in MIDI file!")
                return
            
            grouped_events = []
            current_group = []
            current_tick = None
            tolerance = 2
            
            for tick, note, velocity, track in note_events:
                if current_tick is None or abs(tick - current_tick) <= tolerance:
                    current_group.append((tick, note, velocity, track))
                    current_tick = tick
                else:
                    if current_group:
                        grouped_events.append((current_tick, current_group))
                    current_group = [(tick, note, velocity, track)]
                    current_tick = tick
            
            if current_group:
                grouped_events.append((current_tick, current_group))
            
            current_tempo = 500000
            if tempo_events:
                current_tempo = tempo_events[0][1]
            
            print("Starting in 3 seconds...")
            for countdown in range(3, 0, -1):
                print(f"{countdown}...")
                time_module.sleep(1)
            print("Go!\n")
            
            def ticks_to_seconds(tick_count, tempo_value):
                return (tick_count * tempo_value) / (midi_file.ticks_per_beat * 1000000.0)
            
            start_time = time_module.time()
            real_time_elapsed = 0.0
            last_tick = 0
            tempo_idx = 0
            events_played = 0
            notes_mapped = 0
            total_notes = 0
            
            for event_tick, note_group in grouped_events:
                if self.should_stop:
                    break
                
                while (tempo_idx < len(tempo_events) and 
                       tempo_events[tempo_idx][0] <= event_tick):
                    
                    change_tick, new_tempo = tempo_events[tempo_idx]
                    
                    if change_tick > last_tick:
                        tick_diff = change_tick - last_tick
                        real_time_elapsed += ticks_to_seconds(tick_diff, current_tempo)
                        last_tick = change_tick
                    
                    current_tempo = new_tempo
                    tempo_idx += 1
                
                if event_tick > last_tick:
                    tick_diff = event_tick - last_tick
                    real_time_elapsed += ticks_to_seconds(tick_diff, current_tempo)
                    last_tick = event_tick
                
                target_time = real_time_elapsed / speed_multiplier
                elapsed_time = time_module.time() - start_time
                sleep_time = target_time - elapsed_time
                
                if sleep_time > 0:
                    time_module.sleep(sleep_time)
                
                playable_keys = []
                note_descriptions = []
                note_velocities = []
                
                for _, note_num, velocity, track_num in note_group:
                    mapped_note = self.find_nearest_playable_note(note_num)
                    if mapped_note is not None:
                        key_char = self.note_to_key_map[mapped_note]
                        playable_keys.append(key_char)
                        note_velocities.append(velocity)
                        
                        original_name = self.get_note_name(note_num)
                        if mapped_note == note_num:
                            note_descriptions.append(original_name)
                        else:
                            notes_mapped += 1
                            mapped_name = self.get_note_name(mapped_note)
                            note_descriptions.append(f"{original_name}>{mapped_name}")
                        
                        total_notes += 1
                
                if playable_keys:
                    current_bpm = 60000000 / current_tempo
                    
                    timing_info = []
                    if self.natural_timing:
                        timing_info.append("nat")
                    if self.adaptive_timing:
                        timing_info.append("adapt")
                    if self.simultaneous_chords:
                        timing_info.append("simul")
                    elif self.random_simultaneous_chords:
                        timing_info.append("randsim")
                    if self.randomize_chord_order:
                        timing_info.append("rand")
                    timing_str = " ".join(timing_info) if timing_info else "mech"
                    
                    if len(playable_keys) == 1:
                        note_name = note_descriptions[0]
                        key_name = playable_keys[0]
                        velocity = note_velocities[0]
                        print(f"Note:     {note_name:>39} -> Key:  {key_name:<10} | Vel: {velocity:>3} | BPM: {current_bpm:>3.0f} | {timing_str}")
                        
                        self.queue_key_event(playable_keys[0], 0.008, note_velocities[0])
                    else:
                        chord_notes = "/".join(note_descriptions)
                        if len(chord_notes) > 39:
                            chord_notes = chord_notes[:36] + "..."
                        chord_keys = "".join(playable_keys)
                        avg_velocity = sum(note_velocities) // len(note_velocities)
                        print(f"Chord[{len(playable_keys)}]: {chord_notes:>39} -> Keys: {chord_keys:<10} | Vel: {avg_velocity:>3} | BPM: {current_bpm:>3.0f} | {timing_str}")
                        
                        self.queue_key_event(playable_keys, 0.006, note_velocities)
                    
                    events_played += 1
            
            self.key_queue.join()
            
            print(f"\n{'='*60}")
            print("PLAYBACK COMPLETE")
            print("="*60)
            print(f"Events played    : {events_played} ({total_notes} notes)")
            if tempo_events:
                print(f"Tempo changes    : {len(tempo_events)}")
            if notes_mapped > 0:
                print(f"Octave mapped    : {notes_mapped} notes")
            if self.natural_timing:
                print("Natural timing   : Applied")
            if self.adaptive_timing:
                print("Adaptive chords  : Applied")
            if self.simultaneous_chords:
                print("Simultaneous     : All chords")
            elif self.random_simultaneous_chords:
                print("Simultaneous     : Random chords")
            if self.randomize_chord_order:
                print("Chord randomize  : Enabled")
                
        except KeyboardInterrupt:
            print(f"\n{'='*60}")
            print("PLAYBACK STOPPED BY USER")
            print("="*60)
        except Exception as e:
            print(f"\nError during playback: {e}")
        finally:
            self.is_playing = False
            self.key_queue.put(None)
            if self.key_thread and self.key_thread.is_alive():
                self.key_thread.join(timeout=1.0)


def main():
    parser = argparse.ArgumentParser(
        description="Convert MIDI files to keyboard key presses with enhanced human-like piano playing"
    )
    
    parser.add_argument('midi_file', help='Path to MIDI file')
    parser.add_argument('-c', '--center', type=int, default=60, 
                       help='Center MIDI note (default: 60 = Middle C)')
    parser.add_argument('-s', '--speed', type=float, default=1.0,
                       help='Playback speed multiplier (default: 1.0)')
    parser.add_argument('-n', '--natural', action='store_true',
                       help='Enable full natural timing variations (includes micro-delays, pauses)')
    parser.add_argument('--delay-scale', type=float, default=1.0,
                       help='Scale factor for natural timing delays (default: 1.0, requires --natural)')
    parser.add_argument('--mistake-rate', type=float, default=0.0,
                       help='Human mistake probability 0.0-1.0 (default: 0.0, requires --natural)')
    parser.add_argument('--no-adaptive', action='store_true',
                       help='Disable adaptive timing for complex chords')
    parser.add_argument('--simultaneous', action='store_true',
                       help='Enable truly simultaneous chord playing (all keys pressed at once)')
    parser.add_argument('--random-simultaneous', action='store_true',
                       help='Randomly decide whether to play chords simultaneously or sequentially (very human-like)')
    parser.add_argument('--no-randomize', action='store_true',
                       help='Disable randomization of chord note order')
    parser.add_argument('--test-keys', action='store_true',
                       help='Test keyboard functionality before playing')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.midi_file):
        print(f"Error: File '{args.midi_file}' not found!")
        return
    
    if args.delay_scale != 1.0 and not args.natural:
        print("Error: --delay-scale requires --natural timing to be enabled")
        return
    
    if args.mistake_rate != 0.0 and not args.natural:
        print("Error: --mistake-rate requires --natural timing to be enabled")
        return
    
    if args.mistake_rate < 0.0 or args.mistake_rate > 1.0:
        print("Error: --mistake-rate must be between 0.0 and 1.0")
        return
    
    adaptive_timing = not args.no_adaptive
    simultaneous_chords = args.simultaneous
    random_simultaneous_chords = args.random_simultaneous
    randomize_chord_order = not args.no_randomize
    
    try:
        player = PianoKeyboardPlayer(
            args.center, 
            args.natural, 
            adaptive_timing, 
            args.delay_scale, 
            args.mistake_rate,
            simultaneous_chords,
            randomize_chord_order,
            random_simultaneous_chords
        )
        player.show_piano_layout()
        
        if args.test_keys:
            print("Testing keyboard functionality...")
            
            test_keys = ['1', 'q', 'a', 'z', '!', 'Q']
            for key in test_keys:
                print(f"Testing: '{key}'", end=" ")
                player.keyboard_controller.send_key(key, 0.1)
                time_module.sleep(0.5)
                print("OK")
            
            if simultaneous_chords and hasattr(player.keyboard_controller, 'send_simultaneous_keys'):
                print("Testing simultaneous keys...", end=" ")
                player.keyboard_controller.send_simultaneous_keys(['q', 'w', 'e'], 0.2)
                time_module.sleep(1.0)
                print("OK")
            
            print("\nDid you see the keys appear? (y/n): ", end="")
            response = input().lower().strip()
            if response not in ['y', 'yes']:
                print("Keyboard test failed - check focus and permissions.")
                return
            else:
                print("Keyboard test passed!")
        
        midi_file = player.load_midi_file(args.midi_file)
        if not midi_file:
            return
        
        player.analyze_midi_file(midi_file)
        
        timing_features = []
        if args.natural:
            timing_features.append("natural")
            if args.delay_scale != 1.0:
                timing_features.append(f"{args.delay_scale}x")
            if args.mistake_rate > 0.0:
                timing_features.append(f"{args.mistake_rate:.0%}err")
        if adaptive_timing:
            timing_features.append("adaptive")
        if simultaneous_chords:
            timing_features.append("all_simultaneous")
        elif random_simultaneous_chords:
            timing_features.append("random_simultaneous")
        if randomize_chord_order:
            timing_features.append("randomized")
        if not timing_features:
            timing_features.append("mechanical")
        
        timing_desc = "+".join(timing_features)
        speed_desc = f"{args.speed}x speed" if args.speed != 1.0 else "normal speed"
        
        print(f"\nReady to play: {timing_desc} timing, {speed_desc}")
        print("Make sure your target application is focused!")
        response = input("Continue? (y/n): ").lower().strip()
        if response not in ['y', 'yes', '']:
            print("Cancelled.")
            return
        
        player.play_midi_file(midi_file, args.speed)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
