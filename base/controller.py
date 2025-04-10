import struct

class RCController:
    def __init__(self):
        self.btn_names = ["R1", "L1", "start", "select", "R2", "L2", "F1", "F2",
                          "A", "B", "X", "Y", "up", "right", "down", "left"]

        self.data = {}
        self.data.update({name: False for name in self.btn_names})
        self.data.update({name + "_psd": False for name in self.btn_names})
        self.data.update({"lx": 0.0, "rx": 0.0, "ry": 0.0, "L2": 0.0, "ly": 0.0})

    def update(self, raw_data):
        raw_data_bytes = bytes(raw_data)
        btn_value = struct.unpack('<H', raw_data_bytes[2:4])[0]
        btn_bits = format(btn_value, '016b')

        lx, rx, ry, L2, ly = struct.unpack('<5f', raw_data_bytes[4:24])

        for i, name in enumerate(self.btn_names):
            btn_sts = btn_bits[15 - i] == '1'
            btn_psd = (btn_sts and not self.data[name]) or self.data[name + "_psd"]
            self.data[name + "_psd"] = btn_psd
            self.data[name] = btn_sts

        self.data.update({"lx": lx, "rx": rx, "ry": ry, "L2": L2, "ly": ly})
