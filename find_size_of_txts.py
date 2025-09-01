from azure.storage.blob import ContainerClient

# your connection string
connection_string = "DefaultEndpointsProtocol=https;AccountName=dcgraphrag;AccountKey=QunE9krZp3j8ia+EMrI5s8Cc7jMLFKVqnFL5fOlrhmc0hswuOu+ydhMFeipx0Y+GZHoSGZ0U+XqE+ASt76qW7w==;EndpointSuffix=core.windows.net"

# your container
container_name = "law-cases"

# create client
container = ContainerClient.from_connection_string(conn_str=connection_string, container_name=container_name)

total_size = 0
count = 0

# iterate through all blobs in the container (recursively covers nested paths)
for blob in container.list_blobs():
    if blob.name.lower().endswith(".txt"):
        total_size += blob.size
        count += 1

print(f"Found {count} .txt files")
print(f"Total size: {total_size:,} bytes")
print(f"≈ {total_size/1024/1024:.2f} MB")
print(f"≈ {total_size/1024/1024/1024:.2f} GB")
