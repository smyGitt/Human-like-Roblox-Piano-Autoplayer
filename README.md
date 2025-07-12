# piano-midi-autoplayer
translates .mid (midi) files to keystrokes, designed to work with ROBLOX pianos. Includes random "humanization" effects to imitate the human behind the instrument.

main.py will play the notes with duration, which will hold notes.
noDuration.py, as you can tell from the name, will only press the note for a very short time.
# How do you run it?
You need to run this in a command prompt. After you navigate to where the .py files are, whichever you chose to run, type for example:

`python main.py --help` 

(of course, you may replace main.py with noDuration.py, or any in the backup if desired)

this will show you all the available flags and formatting.

remember that it accepts .mid files only. it works best with piano-only .mid, but I've seen it work with mixed instruments.

# External Libraries
you might need to install a few python libraries though. You will see that main.py imports:

- import argparse
- import mido
- import time
- import threading
- import sys
- import heapq
- import signal
- import random
- import math
- from dataclasses import dataclass
- from typing import Dict, List, Set, Tuple, Optional
- from pynput.keyboard import Key, Controller

and it may differ between the .py files. Please check if you have these installed.
