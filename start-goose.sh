#!/bin/bash
# Launch Goose desktop app with macOS seatbelt sandboxing enabled

export GOOSE_SANDBOX=true
export MLX_LOCAL_API_KEY=not-needed

/Applications/Goose.app/Contents/MacOS/Goose &
