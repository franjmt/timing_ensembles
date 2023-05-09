import time
from PyMata.pymata import PyMata

board_opto = PyMata('COM10', False, False)
board_opto.set_sampling_interval(1)

trigger_in_pin_1 = 7                        #37 --> short
trigger_in_pin_2 = 4                        #35 --> long
opto_out_pin = 8

board_opto.set_pin_mode(opto_out_pin, board_opto.OUTPUT, board_opto.DIGITAL)

def SendTrigger(duration, pin):
    board_opto.digital_write(pin, 1)
    time.sleep(duration/1000)
    board_opto.digital_write(pin, 0)

def read_trigger_1(data):
    global opto_out_pin
    signal = data[2]
    if signal:
        print('short delay trigger received')
        # print('trigger received')
        SendTrigger(490, opto_out_pin)

    board_opto.set_digital_latch(trigger_in_pin_1, board_opto.DIGITAL_LATCH_HIGH, read_trigger_1)
board_opto.set_pin_mode(trigger_in_pin_1, board_opto.INPUT, board_opto.DIGITAL)
board_opto.set_digital_latch(trigger_in_pin_1, board_opto.DIGITAL_LATCH_HIGH, read_trigger_1)

def read_trigger_2(data):
    global opto_out_pin
    signal = data[2]
    if signal:
        print('long delay trigger received')
        # print('trigger received')
        SendTrigger(2050, opto_out_pin)

    board_opto.set_digital_latch(trigger_in_pin_2, board_opto.DIGITAL_LATCH_HIGH, read_trigger_2)
board_opto.set_pin_mode(trigger_in_pin_2, board_opto.INPUT, board_opto.DIGITAL)
board_opto.set_digital_latch(trigger_in_pin_2, board_opto.DIGITAL_LATCH_HIGH, read_trigger_2)

while True:
    continue
