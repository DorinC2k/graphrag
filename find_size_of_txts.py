from azure.storage.blob import ContainerClient
from azure.core.exceptions import ResourceNotFoundError

# your connection string
connection_string = "DefaultEndpointsProtocol=https;AccountName=dcgraphrag;AccountKey=QunE9krZp3j8ia+EMrI5s8Cc7jMLFKVqnFL5fOlrhmc0hswuOu+ydhMFeipx0Y+GZHoSGZ0U+XqE+ASt76qW7w==;EndpointSuffix=core.windows.net"

# your container
container_name = "law-cases"

# prefix you want to search (works with or without a leading slash)
directory_prefix = "/md/csj/civil/2025"

# --- normalize prefix ---
directory_prefix = (directory_prefix or "").lstrip("/")
if directory_prefix and not directory_prefix.endswith("/"):
    directory_prefix += "/"

# create client
container = ContainerClient.from_connection_string(conn_str=connection_string, container_name=container_name)

def show_children(prefix: str, limit: int = 50):
    """
    Print child 'directories' (virtual prefixes) and a few blob names under the given prefix.
    This helps verify whether the prefix exists and how names are structured.
    """
    print(f"\n[Diag] Listing children under: '{prefix or '(root)'}'")
    try:
        # List virtual subfolders first (delimiter='/')
        subdirs = []
        for item in container.walk_blobs(name_starts_with=prefix, delimiter="/"):
            # walk_blobs yields BlobPrefix (has .name) and BlobProperties (has .name, .size, etc.)
            if hasattr(item, "prefix"):  # some SDK versions expose .prefix
                subdirs.append(item.prefix)
            elif hasattr(item, "name") and item.name.endswith("/"):
                subdirs.append(item.name)
            # Stop if too many
            if len(subdirs) >= 50:
                break

        if subdirs:
            print("  Subdirectories:")
            for d in subdirs[:50]:
                print("   -", d)
        else:
            print("  (No subdirectories found here)")

        # Show a few blob names (any extension) directly under prefix
        print("\n  Sample blobs (first few):")
        shown = 0
        for blob in container.list_blobs(name_starts_with=prefix):
            print("   -", blob.name)
            shown += 1
            if shown >= limit:
                break
        if shown == 0:
            print("   (No blobs found under this prefix)")
    except ResourceNotFoundError:
        print("  [Error] Container or path not found.")

def count_txt(prefix: str):
    total_size = 0
    count = 0
    for blob in container.list_blobs(name_starts_with=prefix):
        # lower() handles .TXT/.Txt, etc.
        if blob.name.lower().endswith(".txt"):
            total_size += blob.size or 0
            count += 1
    return count, total_size

print(f"Searching in directory: '{directory_prefix or '(root)'}'")

count, total_size = count_txt(directory_prefix)

print(f"Found {count} .txt files")
print(f"Total size: {total_size:,} bytes")
print(f"≈ {total_size/1024/1024:.2f} MB")
print(f"≈ {total_size/1024/1024/1024:.2f} GB")

# If nothing found, help you navigate to the correct spot
if count == 0:
    show_children(directory_prefix)           # what exists right under that prefix?
    # also try the parent directory to spot a small typo like "civl" vs "civil"
    parent = directory_prefix.rstrip("/")
    parent = parent[:parent.rfind("/")+1] if "/" in parent else ""
    if parent != directory_prefix:
        show_children(parent)
