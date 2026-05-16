import json
import os
from pathlib import Path
from datetime import datetime
from typing import Literal
from dotenv import load_dotenv

from google import genai
from google.genai import types

load_dotenv()
API_KEY = os.getenv('GEMINI_API_KEY')
if not API_KEY:
    raise ValueError('GEMINI_API_KEY não encontrada no .env')

client = genai.Client(api_key=API_KEY)

BASE_DIR = Path(__file__).parent

TIPOS_MIME = {
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.png': 'image/png',
    '.webp': 'image/webp'
}

def carregar_conteudo(arquivo:Path, tipo: Literal['texto', 'imagem']):
    if tipo == 'texto':
        return arquivo.read_text(encoding='utf-8')

    with open(arquivo, 'rb') as f:
        imagem_bytes = f.read()

    mime_type = TIPOS_MIME.get(arquivo.suffix.lower(), 'image/jpeg')
    return types.Part.from_bytes(data=imagem_bytes, mime_type=mime_type)

def gerar_embedding_conteudo(conteudo: str | types.Part, tipo: str) -> list[float]:
    response: types.EmbedContentResponse = client.models.embed_content(
        model='gemini-embedding-2',
        contents=conteudo,
        config=types.EmbedContentConfig(task_type='SEMANTIC_SIMILARITY')
    )
    return response.embeddings[0].values

def gerar_embedding_arquivo(arquivo: Path, tipo: Literal['texto', 'imagem'], origem: Path) -> dict:
    try:
        print(f'  Processando: {arquivo.name}...', end=' ', flush=True)
        conteudo = carregar_conteudo(arquivo, tipo)
        vetor = gerar_embedding_conteudo(conteudo, tipo)

        embedding = {
            'arquivo': arquivo.name,
            'tipo': tipo,
            'caminho_relativo': str(arquivo.relative_to(origem)),
            'timestamp': datetime.now().isoformat(),
            'modelo': 'gemini-embedding-2',
            'dimensoes': len(vetor),
            'embedding': vetor
        }

        print(f'✓ ({len(vetor)}d)')
        return embedding

    except Exception as e:
        print(f'✗ Erro: {e}')
        return {
            'arquivo': arquivo.name,
            'erro': str(e),
            'timestamp': datetime.now().isoformat()
        }

def gerar_embeddings_de_json(url_input: str):
    caminho_json = Path(url_input)

    if not caminho_json.exists():
        raise FileNotFoundError(f'Arquivo não encontrado: {caminho_json}')

    # pasta base do json (para resolver caminhos relativos de imagens)
    origem = caminho_json.parent

    with open(caminho_json, 'r', encoding='utf-8') as f:
        dados = json.load(f)

    if not isinstance(dados, list):
        raise ValueError('O JSON deve conter uma lista de registros.')

    total = len(dados)
    print(f'\nProcessando {total} registros...\n')

    for i, item in enumerate(dados, start=1):
        try:
            metadados = item.get('metadados', {})
            tipo = metadados.get('tipo')

            print(f'[{i}/{total}]', end=' ')

            # CASO 1: conteúdo textual direto no JSON
            if item.get('conteudo') is not None:
                conteudo = item['conteudo']

                print(f'Gerando embedding de texto...', end=' ', flush=True)

                vetor = gerar_embedding_conteudo(conteudo, tipo)

                item['embeddings'] = vetor

                print(f'✓ ({len(vetor)}d)')

            # CASO 2: conteúdo vem de um arquivo (imagem)
            else:
                url_arquivo = metadados.get('url')

                if not url_arquivo:
                    raise ValueError(
                        'Registro sem conteúdo e sem metadados.url'
                    )

                caminho_arquivo = origem / url_arquivo

                if not caminho_arquivo.exists():
                    raise FileNotFoundError(
                        f'Arquivo não encontrado: {caminho_arquivo}'
                    )

                print(
                    f'Gerando embedding de arquivo: {caminho_arquivo.name}...',
                    end=' ',
                    flush=True
                )

                conteudo = carregar_conteudo(caminho_arquivo, tipo)
                vetor = gerar_embedding_conteudo(conteudo, tipo)

                item['embeddings'] = vetor

                print(f'✓ ({len(vetor)}d)')

        except Exception as e:
            print(f'✗ Erro: {e}')

            item['erro'] = str(e)
            item['timestamp_erro'] = datetime.now().isoformat()

    # salvar resultado
    # garantir que a pasta embeddings exista
    pasta_embeddings = BASE_DIR / 'embeddings'
    pasta_embeddings.mkdir(exist_ok=True)
    caminho_saida = pasta_embeddings / (
        f'{caminho_json.stem}_embeddings.json'
    )

    with open(caminho_saida, 'w', encoding='utf-8') as f:
        json.dump(dados, f, ensure_ascii=False, indent=4)

    print(f'\nArquivo salvo em: {caminho_saida}')

    return dados

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Informe o arquivo json com os dados de input:')
        print('python main.py caminho/do/arquivo.json')
        sys.exit(1)

    url_input = sys.argv[1]
    gerar_embeddings_de_json(url_input)