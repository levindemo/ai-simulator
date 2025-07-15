export HTTP_PROXY='http://127.0.0.1:1080'
export HTTPS_PROXY='http://127.0.0.1:1080'
export NO_PROXY='localhost,127.0.0.1'
docker build -t gpt2-packaged .
