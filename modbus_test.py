import serial
import struct
import time

class susgrip:
    def __init__(self):
        # --- CẤU HÌNH CỔNG SERIAL ---
        self.PORT = '/dev/ttyUSB0' 
        self.BAUDRATE = 115200 

    def calculate_crc16(self, data: bytes) -> bytes:
        """Tính mã kiểm tra CRC-16 cho giao thức Modbus RTU"""
        crc = 0xFFFF
        for pos in data:
            crc ^= pos
            for _ in range(8):
                if (crc & 1) != 0:
                    crc >>= 1
                    crc ^= 0xA001
                else:
                    crc >>= 1
        # Modbus RTU yêu cầu byte thấp của CRC gửi trước (Little Endian)
        return struct.pack('<H', crc)

    def send_modbus_rtu_frame(self, pos_value):
        # Mở cổng serial động để tránh xung đột và lỗi đóng cổng
        try:
            ser = serial.Serial(
                port=self.PORT,
                baudrate=self.BAUDRATE,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=1
            )
        except Exception as e:
            print(f"Lỗi khi mở cổng {self.PORT}: {e}")
            return

        # --- CẤU TRÚC FRAME MODBUS ---
        slave_id = 0x01          # Slave ID của thiết bị (thường là 1)
        function_code = 0x06     # Function Code 06: Write Single Register
        
        # Set Force
        #==================================================================================#
        register_force = 0x0004 
        force_value = 10
        data_frame_force = struct.pack('>BBHH', slave_id, function_code, register_force, force_value)
        crc_bytes_force = self.calculate_crc16(data_frame_force)
        full_frame_force = data_frame_force + crc_bytes_force
        
        print(f"Đang gửi Frame Modbus (HEX): {' '.join([f'{b:02X}' for b in full_frame_force])}")
        ser.write(full_frame_force)
        
        time.sleep(0.1)
        response = ser.read(8) # Đọc 8 bytes
        if response:
            print(f"Nhận được phản hồi (HEX):  {' '.join([f'{b:02X}' for b in response])}")
        else:
            print("Không nhận được phản hồi từ thiết bị (Timeout).")
            
        #==================================================================================#
        # Set Position
        register_address = 0x0001
        # pos_value = 0
        data_frame_pos = struct.pack('>BBHH', slave_id, function_code, register_address, pos_value)
        crc_bytes_pos = self.calculate_crc16(data_frame_pos)
        full_frame_pos = data_frame_pos + crc_bytes_pos
        
        print(f"Đang gửi Frame Modbus (HEX): {' '.join([f'{b:02X}' for b in full_frame_pos])}")
        ser.write(full_frame_pos)
        
        time.sleep(0.1)
        response = ser.read(8)
        if response:
            print(f"Nhận được phản hồi (HEX):  {' '.join([f'{b:02X}' for b in response])}")
        else:
            print("Không nhận được phản hồi từ thiết bị (Timeout).")
            
        ser.close()

if __name__ == '__main__':
    susgrip().send_modbus_rtu_frame(100)
