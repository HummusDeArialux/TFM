# Use the official Python 3.10 image as the base image
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Install system packages needed for OpenCV
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx

# Create a non-root user named "skinai"
RUN adduser --disabled-password --gecos '' skinai

# Set ownership of the application directory to the non-root user
RUN chown -R skinai:skinai /app

# Switch to the non-root user
USER skinai

# Copy the requirements file into the container at /app
COPY web/tensored-django/requirements.txt /app/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt --user

# Add the user-specific bin directory to the PATH
ENV PATH="/home/skinai/.local/bin:${PATH}"

# Copy the rest of the application code into the container at /app
COPY web/tensored-django/ /app/

# Expose the port your Django app runs on (change to 8000)
EXPOSE 8000

# Run Django migrations and collect static files (adjust these as needed)
RUN python manage.py migrate
RUN python manage.py collectstatic --noinput

# Define environment variable
#ENV DJANGO_SETTINGS_MODULE=tensored_django.settings

# Run Django app when the container launches
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
