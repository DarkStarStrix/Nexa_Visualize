FROM node:18-alpine

LABEL maintainer="Nexa Visualize Contributors"

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install serve for static file hosting
RUN npm install -g serve

# Expose the port serve will run on
EXPOSE 3000

# Set the default command to serve the public directory
CMD ["serve", "-s", "public", "-l", "3000"]
