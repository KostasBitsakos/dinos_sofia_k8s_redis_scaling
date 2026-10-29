# Stage 1: Building the Go application
FROM golang:alpine AS builder

# Install git. Other tools like bash, sed are not needed for the build.
RUN apk add --no-cache git

# Set the working directory in the container
WORKDIR /app

# Copy go mod and sum files
COPY go.mod  ./

# Download all dependencies
RUN go mod download

# Copy the source code into the container
COPY . .

# Build the Go app
RUN CGO_ENABLED=0 GOOS=linux go build -o redis-cluster-go-coordinator .

# Stage 2: Setup the runtime container
FROM alpine:latest  

# Copy the binary from the builder stage
COPY --from=builder /app/redis-cluster-go-coordinator /usr/local/bin/

# Install redis
RUN apk add --no-cache redis

# Copy redis config and entrypoint script
COPY redis.conf /etc/redis.conf
COPY entrypoint.sh /entrypoint.sh

# Make entrypoint.sh executable
RUN chmod +x /entrypoint.sh

# Run entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

