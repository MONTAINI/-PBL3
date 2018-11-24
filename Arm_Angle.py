#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
import Adafruit_PCA9685

pwm = Adafruit_PCA9685.PCA9685()
pwm.set_pwm_freq(60)
pwm.set_pwm(1, 0, 300)
Servo_A=400
Servo_C=400
#time.sleep(1)
class Arm:
    def angle_LR(self,LR):
        print(LR)
        int(LR)
        if LR <= 90 and LR >= -90:
            Servo_A=(LR/0.36)+400+80       #(角度/ステップ角)+中央値+誤差修正
            #time.sleep(0.1)
            pwm.set_pwm(0, 0, int(Servo_A))
        else:
            print("LR Error")
    def angle_UD(self,UD):
        print(UD)
        int(UD)
        if UD <=45 and UD >=- 40:
            Servo_C=(UD/0.36)+400+40
            #time.sleep(0.1)
            pwm.set_pwm(2, 0, int(Servo_C))
        else:
            print("UD Error")
        
