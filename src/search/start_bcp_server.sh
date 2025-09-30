#!/bin/bash

# Startup script for BrowseComp-Plus search server
# Usage examples:
#   ./start_bcp_server.sh bm25 --index-path /path/to/index
#   ./start_bcp_server.sh faiss --index-path "/path/to/*.pkl" --model-name "Qwen/Qwen3-Embedding-0.6B"

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVE_SCRIPT="$SCRIPT_DIR/serve_bcp.py"

# Default values
PORT=${PORT:-8000}
HOST=${HOST:-"0.0.0.0"}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== BrowseComp-Plus Search Server Startup ===${NC}"

# Check if serve_bcp.py exists
if [[ ! -f "$SERVE_SCRIPT" ]]; then
    echo -e "${RED}Error: $SERVE_SCRIPT not found${NC}"
    exit 1
fi

# Check if at least searcher type is provided
if [[ $# -lt 1 ]]; then
    echo -e "${RED}Error: Searcher type is required${NC}"
    echo -e "${YELLOW}Usage: $0 <searcher-type> [additional-args]${NC}"
    echo -e "${YELLOW}Available searcher types: bm25, faiss, reasonir, custom${NC}"
    echo ""
    echo -e "${YELLOW}Examples:${NC}"
    echo -e "  $0 bm25 --index-path /path/to/lucene/index"
    echo -e "  $0 faiss --index-path '/path/to/*.pkl' --model-name 'Qwen/Qwen3-Embedding-0.6B'"
    echo -e "  $0 reasonir --index-path '/path/to/*.pkl' --model-name 'microsoft/reasonir'"
    echo -e "  $0 faiss --index-path '/path/to/*.pkl' --model-name 'Qwen/Qwen3-Embedding-8B' --snippet-max-tokens 512"
    exit 1
fi

SEARCHER_TYPE="$1"
shift  # Remove searcher type from arguments

echo -e "${GREEN}Searcher Type:${NC} $SEARCHER_TYPE"
echo -e "${GREEN}Host:${NC} $HOST"
echo -e "${GREEN}Port:${NC} $PORT"
echo ""

# Set environment variables for the Python script
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# Common arguments
COMMON_ARGS=(
    "--searcher-type" "$SEARCHER_TYPE"
    "--host" "$HOST"
    "--port" "$PORT"
)

# Add any additional arguments passed to the script
ADDITIONAL_ARGS=("$@")

echo -e "${BLUE}Starting server with command:${NC}"
echo "python3 $SERVE_SCRIPT ${COMMON_ARGS[*]} ${ADDITIONAL_ARGS[*]}"
echo ""

# Start the server
exec python3 "$SERVE_SCRIPT" "${COMMON_ARGS[@]}" "${ADDITIONAL_ARGS[@]}"
