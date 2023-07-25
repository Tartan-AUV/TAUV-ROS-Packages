# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.25

# compile C with /Users/glebryabtsev/.platformio/packages/toolchain-gccarmnoneeabi-teensy/bin/arm-none-eabi-gcc
# compile CXX with /Users/glebryabtsev/.platformio/packages/toolchain-gccarmnoneeabi-teensy/bin/arm-none-eabi-g++
C_DEFINES = -DARDUINO=10805 -DARDUINO_TEENSY41 -DCORE_TEENSY -DF_CPU=600000000 -DLAYOUT_US_ENGLISH -DPLATFORMIO=60109 -DTEENSYDUINO=158 -DUSB_SERIAL -D__IMXRT1062__

C_INCLUDES = -I/Users/glebryabtsev/Dev/tauv_ws/src/TAUV-ROS-Packages/firmware/modem/include -I/Users/glebryabtsev/Dev/tauv_ws/src/TAUV-ROS-Packages/firmware/modem/src -I/Users/glebryabtsev/Dev/tauv_ws/src/TAUV-ROS-Packages/firmware/modem/.pio/libdeps/teensy41/TeensyTimerTool/src -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/cores/teensy4 -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/ADC -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/AccelStepper/src -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/Adafruit_NeoPixel -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/Adafruit_STMPE610 -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/Adafruit_VS1053 -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/Adafruit_nRF8001 -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/Adafruit_nRF8001/utility -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/AltSoftSerial -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/Artnet -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/Audio -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/Audio/utility -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/Bounce -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/Bounce2/src -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/CapacitiveSensor -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/CryptoAccel/src -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/DS1307RTC -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/DmxSimple -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/DogLcd -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/EEPROM -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/EasyTransfer/src -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/EasyTransferI2C/src -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/Encoder -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/Encoder/utility -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/Entropy -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/Ethernet/src -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/FNET/src -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/FastCRC -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/FastLED/src -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/FlexCAN -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/FlexCAN_T4 -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/FlexIO_t4/src -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/FlexiTimer2 -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/FreqCount -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/FreqMeasure -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/FreqMeasureMulti -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/FrequencyTimer2 -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/ILI9341_t3 -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/ILI9488_t3/src -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/IRremote/src -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/Keypad/src -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/LedControl/src -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/LedDisplay -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/LiquidCrystal/src -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/LiquidCrystalFast -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/LittleFS/src -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/LowPower -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/MFRC522/src -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/MIDI/src -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/Metro -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/MsTimer2 -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/NXPMotionSense -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/NXPMotionSense/utility -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/NativeEthernet/src -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/OSC -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/OctoWS2811 -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/OneWire -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/PS2Keyboard -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/PS2Keyboard/utility -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/PWMServo -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/Ping -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/PulsePosition -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/QuadEncoder -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/RA8875/src -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/RadioHead -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/ResponsiveAnalogRead/src -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/SD/src -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/SPI -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/SPIFlash -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/ST7735_t3/src -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/SdFat/src -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/SerialFlash -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/Servo -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/ShiftPWM -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/Snooze/src -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/SoftPWM -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/SoftwareSerial -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/TFT_ILI9163C -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/Talkie -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/TeensyThreads -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/Time -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/TimeAlarms -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/TimerOne -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/TimerThree -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/TinyGPS -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/Tlc5940 -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/TouchScreen -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/USBHost_t36 -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/USBHost_t36/utility -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/UTFT -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/VirtualWire -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/WS2812Serial -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/Wire -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/Wire/utility -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/XBee -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/XPT2046_Touchscreen -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/i2c_t3 -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/ks0108 -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/ssd1351 -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/x10 -I/Users/glebryabtsev/.platformio/packages/toolchain-gccarmnoneeabi-teensy/arm-none-eabi/include/c++/11.3.1 -I/Users/glebryabtsev/.platformio/packages/toolchain-gccarmnoneeabi-teensy/arm-none-eabi/include/c++/11.3.1/arm-none-eabi -I/Users/glebryabtsev/.platformio/packages/toolchain-gccarmnoneeabi-teensy/lib/gcc/arm-none-eabi/11.3.1/include -I/Users/glebryabtsev/.platformio/packages/toolchain-gccarmnoneeabi-teensy/lib/gcc/arm-none-eabi/11.3.1/include-fixed -I/Users/glebryabtsev/.platformio/packages/toolchain-gccarmnoneeabi-teensy/arm-none-eabi/include

C_FLAGS = -Wall;-ffunction-sections;-fdata-sections;-mthumb;-mcpu=cortex-m7;-nostdlib;-mfloat-abi=hard;-mfpu=fpv5-d16;-O2 -fcolor-diagnostics

CXX_DEFINES = -DARDUINO=10805 -DARDUINO_TEENSY41 -DCORE_TEENSY -DF_CPU=600000000 -DLAYOUT_US_ENGLISH -DPLATFORMIO=60109 -DTEENSYDUINO=158 -DUSB_SERIAL -D__IMXRT1062__

CXX_INCLUDES = -I/Users/glebryabtsev/Dev/tauv_ws/src/TAUV-ROS-Packages/firmware/modem/include -I/Users/glebryabtsev/Dev/tauv_ws/src/TAUV-ROS-Packages/firmware/modem/src -I/Users/glebryabtsev/Dev/tauv_ws/src/TAUV-ROS-Packages/firmware/modem/.pio/libdeps/teensy41/TeensyTimerTool/src -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/cores/teensy4 -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/ADC -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/AccelStepper/src -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/Adafruit_NeoPixel -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/Adafruit_STMPE610 -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/Adafruit_VS1053 -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/Adafruit_nRF8001 -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/Adafruit_nRF8001/utility -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/AltSoftSerial -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/Artnet -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/Audio -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/Audio/utility -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/Bounce -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/Bounce2/src -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/CapacitiveSensor -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/CryptoAccel/src -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/DS1307RTC -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/DmxSimple -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/DogLcd -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/EEPROM -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/EasyTransfer/src -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/EasyTransferI2C/src -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/Encoder -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/Encoder/utility -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/Entropy -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/Ethernet/src -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/FNET/src -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/FastCRC -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/FastLED/src -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/FlexCAN -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/FlexCAN_T4 -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/FlexIO_t4/src -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/FlexiTimer2 -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/FreqCount -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/FreqMeasure -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/FreqMeasureMulti -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/FrequencyTimer2 -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/ILI9341_t3 -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/ILI9488_t3/src -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/IRremote/src -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/Keypad/src -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/LedControl/src -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/LedDisplay -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/LiquidCrystal/src -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/LiquidCrystalFast -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/LittleFS/src -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/LowPower -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/MFRC522/src -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/MIDI/src -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/Metro -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/MsTimer2 -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/NXPMotionSense -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/NXPMotionSense/utility -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/NativeEthernet/src -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/OSC -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/OctoWS2811 -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/OneWire -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/PS2Keyboard -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/PS2Keyboard/utility -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/PWMServo -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/Ping -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/PulsePosition -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/QuadEncoder -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/RA8875/src -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/RadioHead -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/ResponsiveAnalogRead/src -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/SD/src -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/SPI -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/SPIFlash -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/ST7735_t3/src -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/SdFat/src -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/SerialFlash -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/Servo -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/ShiftPWM -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/Snooze/src -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/SoftPWM -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/SoftwareSerial -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/TFT_ILI9163C -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/Talkie -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/TeensyThreads -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/Time -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/TimeAlarms -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/TimerOne -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/TimerThree -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/TinyGPS -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/Tlc5940 -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/TouchScreen -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/USBHost_t36 -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/USBHost_t36/utility -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/UTFT -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/VirtualWire -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/WS2812Serial -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/Wire -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/Wire/utility -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/XBee -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/XPT2046_Touchscreen -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/i2c_t3 -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/ks0108 -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/ssd1351 -I/Users/glebryabtsev/.platformio/packages/framework-arduinoteensy/libraries/x10 -I/Users/glebryabtsev/.platformio/packages/toolchain-gccarmnoneeabi-teensy/arm-none-eabi/include/c++/11.3.1 -I/Users/glebryabtsev/.platformio/packages/toolchain-gccarmnoneeabi-teensy/arm-none-eabi/include/c++/11.3.1/arm-none-eabi -I/Users/glebryabtsev/.platformio/packages/toolchain-gccarmnoneeabi-teensy/lib/gcc/arm-none-eabi/11.3.1/include -I/Users/glebryabtsev/.platformio/packages/toolchain-gccarmnoneeabi-teensy/lib/gcc/arm-none-eabi/11.3.1/include-fixed -I/Users/glebryabtsev/.platformio/packages/toolchain-gccarmnoneeabi-teensy/arm-none-eabi/include

CXX_FLAGS = -fno-exceptions;-felide-constructors;-fno-rtti;-std=gnu++14;-Wno-error=narrowing;-fpermissive;-fno-threadsafe-statics;-Wall;-ffunction-sections;-fdata-sections;-mthumb;-mcpu=cortex-m7;-nostdlib;-mfloat-abi=hard;-mfpu=fpv5-d16;-O2 -fcolor-diagnostics -std=gnu++14
