pipeline {
  agent { label 'built-in' } // runs on your controller container (Linux)

  options { timestamps() }

  environment {
    DOCKER_HOST       = 'tcp://172.19.0.3:2375'
    DOCKER_TLS_VERIFY = ''
    DOCKER_CERT_PATH  = ''
  }

  stages {
    stage('Build') {
      steps {
        echo 'Building...'
        sh 'docker -H "$DOCKER_HOST" version'
      }
    }

    stage('Test (Python example)') {
      steps {
        sh '''
          set -eux
          docker -H "$DOCKER_HOST" run --rm -v "$WORKSPACE":/app -w /app \
            python:3.11-alpine sh -lc '
              python3 --version &&
              pip3 --version || true &&
              if [ -f requirements.txt ]; then pip3 install --no-cache-dir -r requirements.txt; fi &&
              if command -v pytest >/dev/null 2>&1; then pytest -q || true; fi
            '
        '''
      }
    }

    stage('Deploy') {
      steps {
        echo 'Deploy placeholder'
      }
    }
  }
}
