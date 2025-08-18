#!/bin/sh
# Start the Azure Cosmos DB Emulator using Docker

docker run \
  --name cosmosdb-emulator \
  --privileged \
  -p 8081:8081 \
  -p 10250-10255:10250-10255 \
  -e AZURE_COSMOS_EMULATOR_PARTITION_COUNT=10 \
  -e AZURE_COSMOS_EMULATOR_ENABLE_DATA_PERSISTENCE=true \
  -e AZURE_COSMOS_EMULATOR_IP_ADDRESS_OVERRIDE=127.0.0.1 \
  --cpus=2 \
  --memory=3g \
  mcr.microsoft.com/cosmosdb/linux/azure-cosmos-emulator
