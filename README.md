# LAFL
Large Array Foveated Listening 

The code base for both the Python script running on the RPI and on a laptop


Things to do
- Fix graph big/little bit mixup
    -- Fixed 2/27/24 (Both need to be set to big/MSB first)
- expand graph to all 12 signals
    - Code on both the pi and laptop
    -- Fixed probably 
        - still needs real test
- Implement FFT code
    - A step for noise reduction
    -- Done for one mic 3/24/24
        - still needs real test
    - common mode rejection
        - Implemented but might not be useful
            - Time will tell
        
- Filtering code -- Cole
    - Filter types
- Difference of phase code -- Casie && Possibly Tyler
    - I don't even know what goes into this
    - also need code to auto align microphone offsets in code
- Signal saving and replay with HW emulator -- Mathew
- Box -- Robert


FOR LATER EXPLORATION
- UI design

- Possible way to speed up code if we need full 50kHz or just more than we have
    - Seems counter intuitive but we use a delay
    - So instead of sending data at a rate of 50khz we save up a seconds worth of data and then send that as a bigger packet
    - This might speed up the system b/c most of the time it taken with the network protocol. 
        - So less protocol = more time


1. Setting up Github Desktop
- I need user names so I can make you a contributor
- Clone from URL https://github.com/MattRSON/LAFL
- Set branch to current task
    - If you try to push to main it will tell you no.
    - Only push to the current task
    - Once the task is done I will move it to main
- Any change in the files will show in github desktop
    - They can then be committed and pushed

2. VS code
- Install python 3.12 from windows store
- Install python extension in vs code
- Install Live Share extension in vs code
- Open LAFL folder in documents/github
- SPI-Network runs on the Rpi
- LaptopPython runs on laptop
- Any test codes that you want to write can be thrown in the TestScripts folder

3. Running Code
- If a terminal is now open within Vs Code hit Terminal/New Terminal at the top
- From this hit the small plus in the top right of the terminal window (Bottom right of the screen)
    - One of these terminals will be used to run code on the laptop
    - The other will be used to run code on the Rpi
- To run the code on the laptop side type 'python LaptopPython.py'
    - 'python' Tells it to run as a python file 
    - 'LaptopPython.py' in the name of the script
    - This will only work if the Rpi code is running
    - To stop the code hit the x on the window
- To run the code on the Rpi side we first need to connect to it
    - in the unused terminal type 'ssh pi@LAFL'
        - This will connect to it as user 'pi' to computer 'LAFL'
    - The password is LAFLTSMCR
    - Then to run the code we need to ender the LAFL folder
    - and lastly type 'sudo python SPI-Network.py'
        - The code will show nothing, this is normal
        - A warning might pop up. In can be ignored
        - To stop the code hit 'ctrl C'

4. Navigating in the command line
- Changing folder
    - 'cd FOLDERNAME' will go to that folder
    - 'cd ..' will go back one folder
    - 'ls' will list the folders and files in a folder
- Running things
    - 'sudo' Gives admin privilege needed for some commands
- Shutting down the pi
    - 'sudo shutdown now' shuts it down so its safe to unplug