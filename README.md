# RAG API Service

A FastAPI-based REST API service that provides Retrieval-Augmented Generation (RAG) functionality for querying professor information using conversational AI with memory and context awareness.

## Overview

This service exposes HTTP endpoints for interactive Q&A about professors, integrating with a RAG pipeline that combines retrieval from vector databases with language model generation. It maintains conversation history and provides contextual responses based on professor-specific data.

## Features

- **RAG Pipeline**: Combines document retrieval with AI-generated responses
- **Conversation Memory**: Maintains chat history for contextual interactions
- **Professor Selection**: Query-specific professor filtering and data retrieval
- **CORS Support**: Configured for cross-origin requests from frontend applications
- **Database Integration**: Retrieves professor lists from PostgreSQL
- **Containerized**: Includes Dockerfile and Jenkins pipeline for automated deployment
