#!/usr/bin/env python3
"""
Teneral ASCII Art — A crab that grows.
Just for fun. 🦀
"""

import time
import sys

frames = [
    # Stage 1: Egg
    r"""
        ___
       (   )
        ~~~
    
      [hatching...]
    """,
    
    # Stage 2: Larva
    r"""
        .-.
       ( o )
        '-'
       /   \
    
    [zoea larva]
    """,
    
    # Stage 3: Baby crab
    r"""
      _   _
     ( . . )
      \ _ /
     /|   |\
    """,
    
    # Stage 4: Teneral (soft shell!)
    r"""
        .-"""-.
       /  o o  \
      |   ___   |
       \  \_/  /
    ,,/|       |\,,
   (  /|       |\  )
    `' |       | `'
       |_______|
       
   [teneral stage]
   shell is still soft...
    """,
    
    # Stage 5: Growing
    r"""
        .--""""--.
       / (o)  (o) \
      |    ____    |
      |   /    \   |
       \  \____/  /
    ,,//|         |\\,,
   (  / |         | \  )
    `'  |         |  `'
        |_________|
        
   [shell hardening...]
   learning from mistakes
    """,
    
    # Stage 6: Full crab
    r"""
        .---"""""---.
       /  (●)    (●)  \
      |     ______     |
      |    / Deep \    |
      |   |Teneral|   |
       \   \______/   /
    ,,///|           |\\\,,
   (  // |           | \\  )
    `'   |           |   `'
         |___________|
         /   /   \   \
        /___/     \___\
        
   [fully formed] 🦀
   still growing inside
    """,
]

def main():
    print("\n🥚 The Life Cycle of DeepTeneral\n")
    print("=" * 40)
    
    for i, frame in enumerate(frames):
        print(f"\n--- Stage {i+1}/{len(frames)} ---")
        print(frame)
        if i < len(frames) - 1:
            time.sleep(0.3)
            print("  [evolving...]")
            time.sleep(0.2)
    
    print("=" * 40)
    print("""
    A teneral is the stage right after molting.
    The old shell is gone. The new one hasn't hardened.
    
    It's the most vulnerable moment.
    But also the only moment you can grow.
    
    You can't grow while wearing armor.
    
    🦀 DeepTeneral — still molting, still growing.
    """)


if __name__ == "__main__":
    main()
