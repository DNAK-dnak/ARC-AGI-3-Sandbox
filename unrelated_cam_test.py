import cv2

# Chọn camera (0 là mặc định)
cap = cv2.VideoCapture(2)

# Thiết lập độ phân giải cực đại (OpenCV sẽ tự chọn mức cao nhất mà camera hỗ trợ)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 10000)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 10000)

# Đọc lại thông số thực tế mà camera đã nhận
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Độ phân giải đang sử dụng: {width} x {height}")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Hiển thị frame gốc (Full feed)
    cv2.imshow('Full HD Camera Feed', frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
