# Listen to color

## Inspiration

http://www.ted.com/talks/neil_harbisson_i_listen_to_color

After listening to the above TED talk I felt inspired to give it a shot myself.
This is the result of a few hours of hacking things together.

## Warning

Turn your audio way down before running this. It doesn't sound nice, additioinal processing is needed to make a better selection of wave forms to superimpose.

## Running

    python colorlisten.py

## Shortcomings / TODO

Audio is choppy. Generating and streaming the waveforms in a separate thread might help. I'm not even sure that this will make the choppiness go away completely. Maybe there's a better library.
