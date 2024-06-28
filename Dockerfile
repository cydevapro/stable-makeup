# Sử dụng một base image chứa Miniconda hoặc Anaconda
FROM continuumio/miniconda3

# Thiết lập thư mục làm việc
WORKDIR /app

# Sao chép mã nguồn của ứng dụng vào Docker image
COPY . .

# Cài đặt các dependencies bằng pip
RUN pip install --no-cache-dir flask pillow

# Cài đặt môi trường biến của Flask
ENV FLASK_APP=app_test.py
ENV FLASK_RUN_HOST=0.0.0.0

# Expose cổng 5000 của Flask
EXPOSE 5000

# Chạy ứng dụng Flask khi container được khởi động
CMD ["flask", "run"]
