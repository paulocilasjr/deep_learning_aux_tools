#!/bin/bash

echo "Available tools:"
echo "1. Embedding Splitter (embedding_splitter.py)"
echo "2. MIL Bag Creator (mil_bag_creator.py)"
echo
echo "Run a tool using: python3 <tool_name> [options]"

exec "$@"
