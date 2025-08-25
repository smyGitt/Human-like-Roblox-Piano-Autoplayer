# Ultimate-Humanised-MIDI-Player-for-ROBLOX, MIDI2Key
MIDI2Key is a Python-based tool that converts MIDI files into keystrokes for ROBLOX piano simulations. It includes humanization effects, automated pedal support, and multiple playback modes for realistic virtual performances. Easy to run via executable or source code, with GUI support included.

<img width="326" height="396" alt="MIDI2Key GUI for ROBLOX Piano" src="https://github.com/user-attachments/assets/b43864f5-80b0-4b6b-8a52-ab943c622029"/>


main.py will play the notes with duration, which will hold notes.
noDuration.py, as you can tell from the name, will only press the note for a very short time.
However, noDuration.py is very unpolished compared to main.py. I highly suggest you use main.py or the .exe from the releases.
# How do you run it?
## If you downloaded the .exe from the releases:
just run the .exe (there probably shouldn't be any malfunctions), or create your own with pyinstaller. I've provided the icon `icon.ico`, so use with this command:

    pyinstaller --onefile --windowed --icon="icon.ico" --add-data="icon.ico;." --name MIDI2Key main.py

  make sure the .ico file and .py file is in the same directory.

## If you don't trust me...
You need to run this in a command prompt. After you navigate to where the .py files are, whichever you chose to run, type for example:

    python main.py
  or, if you are not going to use the main.py with GUI, for whatever reason, navigate to the backup folder, open command prompt there, then enter:

    python final_beforeGUI.py --help

  (of course, you may replace final_beforeGUI.py with noDuration.py, or any in the backup if desired)

this will show you all the available flags and formatting.

remember that it accepts .mid files only. it works best with piano-only .mid, but I've seen it work with mixed instruments.

# Dependencies
you might need to install a few python libraries though. You will see that main.py imports various libraries.

    import mido, time, headpq, threading, random, copy, numpy, sys, dataclasses, import, typing, collections, os, PyQt6  

and it may differ between the .py files. Please check if you have these installed.
