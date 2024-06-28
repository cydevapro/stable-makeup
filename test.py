import ngrok

listener = ngrok.forward(5000, "tcp",authtoken="2YtApGcOINFy3F3oA7T0uxIkIIn_nie3EqfnyfaoyMieAiZC")
# Output ngrok url to console
print(listener.url())