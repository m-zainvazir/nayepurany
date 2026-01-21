# Install screen if not available
sudo apt install screen

# Create a new screen session
screen -S fastapi-app

# Run your app
uvicorn doc_rag:app --host 0.0.0.0 --port 8080

# Detach from screen (app keeps running)
######## Press: Ctrl+A then D

# To reattach later:
screen -r fastapi-app




Method 1: Using nohup (Easiest)

nohup uvicorn doc_rag:app --host 0.0.0.0 --port 8080 > app.log 2>&1 &
# Kill using the PID you saw (3420288)
kill 3420288