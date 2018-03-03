FROM coinstac/coinstac-base

# Set the working directory
WORKDIR /computation

COPY requirements.txt /computation

RUN pip install -r requirements.txt

# Copy the current directory contents into the container
COPY . /computation

# Install any needed packages specified in requirements.txt

