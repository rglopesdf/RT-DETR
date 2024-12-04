#!/bin/bash

# Verifica se o primeiro parâmetro foi passado, caso contrário, usa "."
commit_message=${1:-"."}

# Salva o diretório original
original_dir=$(pwd)

# Muda para o diretório desejado
cd /content/RT-DETR || { echo "Erro: Diretório /content/RT-DETR não encontrado."; exit 1; }

# Executa os comandos git
git add .
git commit -m "$commit_message"
git push origin dev

# Volta ao diretório original
cd "$original_dir" || { echo "Erro: Não foi possível retornar ao diretório original."; exit 1; }

# Confirmação
echo "Commit realizado e retornado ao diretório original."