# SQL-AI-samples

## About this repo

This repo hosts samples meant to help design [AI applications built on data from an Azure SQL Database](https://aka.ms/sql-ai). We illustrate key technical concepts and demonstrate workflows that integrate Azure SQL data with other popular AI application components inside and outside of Azure.
이 저장소는 Azure SQL 데이터베이스의 데이터를 기반으로 AI 애플리케이션을 설계하는 데 도움이 되는 샘플을 호스팅합니다. 우리는 주요 기술 개념을 설명하고 Azure 내부 및 외부의 다른 인기 AI 애플리케이션 구성 요소와 Azure SQL 데이터를 통합하는 워크플로우를 시연합니다.

- [AI Features Samples](#ai-features-samples)
    - [Azure SQL + Azure Cognitive Services](#azure-sql--azure-cognitive-services)
    - [Azure SQL + Azure Promptflow](#azure-sql--azure-promptflow)
    - [Azure SQL + Azure OpenAI](#azure-sql--azure-openai)
    - [Generating SQL for Azure SQL Database using Vanna.AI](#generating-sql-for-azure-sql-database-using-vannaai)
    - [Retrieval Augmented Generation (T-SQL Sample)](#retrieval-augmented-generation-t-sql-sample)
    - [Content Moderation](#content-moderation)
    - [LangChain and Azure SQL Database](#langchain-and-azure-sql-database)
- [End-To-End Samples](#end-to-end-samples)
    - [Similar Content Finder](#similar-content-finder)
    - [Session Conference Assistant](#session-conference-assistant)
    - [Chatbot on your own data with LangChain and Chainlit](#chatbot-on-your-own-data-with-langchain-and-chainlit)
    - [Chatbot on structured and unstructured data with Semantic Kernel](#chatbot-on-structured-and-unstructured-data-with-semantic-kernel)
    - [Azure SQL DB Vectorizer](#azure-sql-db-vectorizer)
    - [SQL Server Database Development using Prompts as T-SQL Development](#sql-server-database-development-using-prompts-as-t-sql-development)
    - [Redis Vector Search Demo Application using ACRE and Cache Prefetching from Azure SQL with Azure Functions](#redis-vector-search-demo-application-using-acre-and-cache-prefetching-from-azure-sql-with-azure-functions)
    - [Similarity Search with FAISS and Azure SQL](#similarity-search-with-faiss-and-azure-sql)
    - [Build your own IVFFlat index with KMeans](#build-your-own-ivfflat-index-with-kmeans)
- [Workshops](#workshops)
    - [Build an AI App GraphQL Endpoint with SQL DB in Fabric​](#build-an-ai-app-graphql-endpoint-with-sql-db-in-fabric​)     

## AI Features Samples

### Azure SQL + Azure Cognitive Services

The [AzureSQL_CogSearch_IntegratedVectorization](https://github.com/Azure-Samples/SQL-AI-samples/blob/main/AzureSQLACSSamples/src/AzureSQL_CogSearch_IntegratedVectorization.ipynb) sample notebook shows a simple AI application that recommends products based on a database of user reviews, using Azure Cognitive Search to store and search the relevant data. It highlights new preview features of Azure Cognitive Search, including automatic chunking and integrated vectorization of user queries.
AazureSQL_CogSearch_IntegratedVectorization 샘플 노트북은 사용자 리뷰 데이터베이스를 기반으로 제품을 추천하는 간단한 AI 애플리케이션을 보여줍니다. 이 애플리케이션은 Azure Cognitive Search를 사용하여 관련 데이터를 저장하고 검색합니다. 또한 자동 청크화 및 사용자 쿼리의 통합 벡터화와 같은 Azure Cognitive Search의 새로운 미리 보기 기능을 강조합니다.

### Azure SQL + Azure Promptflow 

The [AzureSQL_Prompt_Flow](https://github.com/Azure-Samples/SQL-AI-samples/tree/main/AzureSQLPromptFlowSamples) sample shows an E2E example of how to build AI applications with Prompt Flow, Azure Cognitive Search, and your own data in Azure SQL database. It includes instructions on how to index your data with Azure Cognitive Search, a sample Prompt Flow local development that links everything together with Azure OpenAI connections, and also how to create an endpoint of the flow to an Azure ML workspace.
AzureSQL_Prompt_Flow 샘플은 Prompt Flow, Azure Cognitive Search 및 Azure SQL 데이터베이스의 데이터를 사용하여 AI 애플리케이션을 구축하는 E2E(End-to-End) 예제를 보여줍니다. 이 샘플에는 Azure Cognitive Search를 사용하여 데이터를 인덱싱하는 방법, Azure OpenAI 연결로 모든 것을 연결하는 Prompt Flow 로컬 개발 샘플, 그리고 Azure ML 작업 공간으로 흐름의 엔드포인트를 생성하는 방법에 대한 지침이 포함되어 있습니다.

### Azure SQL + Azure OpenAI 

This example shows how to use Azure OpenAI from Azure SQL database to get the vector embeddings of any chosen text, and then calculate the cosine similarity against the Wikipedia articles (for which vector embeddings have been already calculated,) to find the articles that covers topics that are close - or similar - to the provided text.
이 예제는 Azure SQL 데이터베이스에서 Azure OpenAI를 사용하여 선택한 텍스트의 벡터 임베딩을 가져오는 방법을 보여줍니다. 그런 다음 이미 벡터 임베딩이 계산된 위키피디아 기사와의 코사인 유사성을 계산하여 제공된 텍스트와 가까운 주제를 다루는 기사를 찾습니다.

https://github.com/Azure-Samples/azure-sql-db-openai

### Generating SQL for Azure SQL Database using Vanna.AI
This notebook runs through the process of using the `vanna` Python package to generate SQL using AI (RAG + LLMs) including connecting to a database and training.
이 노트북은 vanna Python 패키지를 사용하여 AI (RAG + LLMs)를 통해 SQL을 생성하는 과정을 설명합니다. 여기에는 데이터베이스에 연결하고 학습하는 과정이 포함됩니다.
https://github.com/Azure-Samples/SQL-AI-samples/blob/main/AzureSQLDatabase/Vanna.ai/vanna_and_sql.ipynb

### Retrieval Augmented Generation (T-SQL Sample)

In this repo you will find a step-by-step guide on how to use Azure SQL Database to do Retrieval Augmented Generation (RAG) using the data you have in Azure SQL and integrating with OpenAI, directly from the Azure SQL database itself. You'll be able to ask queries in natural language and get answers from the OpenAI GPT model, using the data you have in Azure SQL Database.
이 저장소에서는 Azure SQL 데이터베이스를 사용하여 Azure SQL에 있는 데이터를 활용하고 OpenAI와 통합하여 Retrieval Augmented Generation (RAG)을 수행하는 방법에 대한 단계별 가이드를 제공합니다. 자연어로 쿼리를 입력하면 Azure SQL 데이터베이스에 있는 데이터를 사용하여 OpenAI GPT 모델로부터 답변을 받을 수 있습니다.
https://github.com/Azure-Samples/azure-sql-db-chatbot

### Content Moderation

In this folder are two T-SQL scripts that call Azure OpenAI Content Safety and Language AI. The Content Safety example will analyze a text string and return a severity in four categories: violence, sexual, self-harm, and hate. The Language AI script will analyze text and return what PII it found, what category of PII it is, and redact the results to obfuscate the PII in the original text string.
이 폴더에는 Azure OpenAI의 콘텐츠 안전 및 언어 AI를 호출하는 두 개의 T-SQL 스크립트가 포함되어 있습니다. 콘텐츠 안전 예제는 텍스트 문자열을 분석하고 폭력, 성적, 자해 및 증오의 네 가지 범주에서 심각도를 반환합니다. 언어 AI 스크립트는 텍스트를 분석하여 발견된 개인 식별 정보(PII), 해당 PII의 범주를 반환하고, 원본 텍스트 문자열에서 PII를 가리기 위해 결과를 수정합니다.
https://github.com/Azure-Samples/SQL-AI-samples/tree/main/AzureSQLDatabase/ContentModeration

### LangChain and Azure SQL Database

This folder contains 2 python notebooks that use LangChain to create a NL2SQL agent against an Azure SQL Database. The notebooks use either Azure OpenAI or OpenAI for the LLM. To get started immedietly, you can create a codespace on this repository, use the terminal to change to the LangChain directory and follow one of the notebooks.
이 폴더에는 Azure SQL 데이터베이스에 대해 NL2SQL 에이전트를 생성하기 위해 LangChain을 사용하는 두 개의 Python 노트북이 포함되어 있습니다. 노트북은 LLM에 대해 Azure OpenAI 또는 OpenAI를 사용합니다. 즉시 시작하려면 이 저장소에서 코드 스페이스를 생성하고 터미널에서 LangChain 디렉토리로 변경한 후 노트북 중 하나를 따라 할 수 있습니다.
https://github.com/Azure-Samples/SQL-AI-samples/tree/main/AzureSQLDatabase/LangChain

You can also use the Getting Started samples available on LangChain website, but using Azure SQL:
또한 LangChain 웹사이트에서 제공하는 시작하기 샘플을 사용할 수 있지만, Azure SQL을 사용하는 방법도 있습니다.
https://github.com/Azure-Samples/azure-sql-langchain

## End-To-End Samples

### Similar Content Finder

OpenAI embeddings, and thus vectors, can be used to perform similarity search and create solution that provide customer with a better user experience, better search results and in general a more natural way to find relevant data in a reference dataset. Due to ability to provide an answer even when search request do not perfectly match a given content, similary search is ideal for creating recommenders. A fully working end-to-end sample is available here: 

https://github.com/Azure-Samples/azure-sql-db-session-recommender

###  Session Conference Assistant

This sample demonstrates how to build a session assistant using Jamstack, Retrieval Augmented Generation (RAG) and Event-Driven architecture, using Azure SQL DB to store and search vectors embeddings generated using OpenAI. The solution is built using Azure Static Web Apps, Azure Functions, Azure SQL Database, and Azure OpenAI. A fully working, production ready, version of this sample, that has been used at VS Live conferences, is available here: https://ai.microsofthq.vslive.com/

https://github.com/azure-samples/azure-sql-db-session-recommender-v2

### Chatbot on your own data with LangChain and Chainlit

Sample RAG pattern, with full UX, using Azure SQL DB, Langchain and Chainlit as demonstrated in the [#RAGHack](https://github.com/microsoft/RAG_Hack) conference. Full details and video recording available here: [RAG on Azure SQL Server](https://github.com/microsoft/RAG_Hack/discussions/53).

https://github.com/Azure-Samples/azure-sql-db-rag-langchain-chainlit

### Chatbot on structured and unstructured data with Semantic Kernel

A chatbot that can answer using RAG and using SQL Queries to answer any question you may want to ask it, be it on unstructured data (eg: what is the common issue raised for product XYZ) or on structured data (eg: how many customers from Canada called the support line?). Built using Semantic Kernel.

https://github.com/Azure-Samples/azure-sql-db-chat-sk

### Azure SQL DB Vectorizer

Quickly chunk text and generate embeddings at scale with data from Azure SQL. 

https://github.com/Azure-Samples/azure-sql-db-vectorizer

###  SQL Server Database Development using Prompts as T-SQL Development

In this notebook, we will learn how to use prompts as a way to develop and test Transact-SQL (T-SQL) code for SQL Server databases. Prompts are natural language requests that can be converted into T-SQL statements by using Generative AI models, such as GPT-4. This can help us write code faster, easier, and more accurately, as well as learn from the generated code examples.

https://github.com/Azure-Samples/SQL-AI-samples/tree/main/AzureSQLDatabase/Prompt-Based%20T-SQL%20Database%20Development

### Redis Vector Search Demo Application using ACRE and Cache Prefetching from Azure SQL with Azure Functions

We based this project from our Product Search Demo which showcase how to use Redis as a Vector Db. We modified the demo by adding a Cache Prefetching pattern from Azure SQL to ACRE using Azure Functions. The Azure Function uses a SQL Trigger that will trigger for any updates that happen in the table.

https://github.com/AzureSQLDB/redis-azure-ai-demo

### Similarity Search with FAISS and Azure SQL

This contains Python notebooks that integrate Azure SQL Database with FAISS for efficient similarity search. The notebooks demonstrate how to store and query data in Azure SQL, leveraging FAISS for fast similarity search. We will be demonstrating it with Wikipedia movie plots data stored in Azure SQL. We’ll encode these movie plots into dense vectors using a pre-trained model and then create a FAISS index to perform similarity searches.
Learn more in the detail blog and video: https://aka.ms/azuresql-faiss

https://github.com/Azure-Samples/SQL-AI-samples/tree/main/AzureSQLFaiss

### Build your own IVFFlat index with KMeans

This sample demonstrates how to perform Approximate Nearest Neighbor (ANN) search on a vector column in Azure SQL DB using KMeans clustering, a technique known as IVFFlat or Cell-Probing. The project utilizes the SciKit Learn library for clustering, storing results in a SQL DB table to facilitate ANN search. This approach is beneficial for speeding up vector searches in Azure SQL DB. 

## Workshops

### Build an AI App GraphQL Endpoint with SQL DB in Fabric​

This lab will guide you through creating a set of GraphQL RAG application APIs that use relational data, Azure OpenAI, and SQL DB in Fabric.

https://github.com/Azure-Samples/sql-in-fabric-ai-embeddings-workshop

## Getting started

See the description in each sample for instructions (projects will have either a README file or instructions in the notebooks themselves.)

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
