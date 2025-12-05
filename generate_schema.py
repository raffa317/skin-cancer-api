import json
from api import app
from fastapi.openapi.utils import get_openapi

# Generate the schema
schema = get_openapi(
    title=app.title,
    version=app.version,
    openapi_version=app.openapi_version,
    description=app.description,
    routes=app.routes,
)

# Fix for FlutterFlow: Ensure the file upload is correctly typed for multipart
# FastAPI generates it correctly, but sometimes explicit tweaking helps if FF is picky.
# For now, standard generation should work.

with open("openapi.json", "w") as f:
    json.dump(schema, f, indent=2)
print("openapi.json generated successfully.")
