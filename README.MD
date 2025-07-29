# ADO MCP Agent

Un agente inteligente basado en LLM que integra Azure DevOps (ADO) utilizando Model Context Protocol (MCP) para proporcionar análisis automatizado, insights y asistencia en la gestión de proyectos de desarrollo.

## 🎯 Objetivo del Proyecto

Este Proof of Concept (POC) demuestra cómo un agente LLM puede interactuar de manera inteligente con Azure DevOps para:

- Analizar el estado de sprints y proyectos
- Generar reportes automatizados de progreso
- Proporcionar insights sobre work items, pull requests y builds
- Asistir en la toma de decisiones de gestión de proyectos
- Automatizar tareas repetitivas de seguimiento y análisis

## 🏗️ Arquitectura

El proyecto utiliza una arquitectura modular que combina:

- **LangChain & LangGraph**: Para orquestación del agente y flujos de trabajo
- **Model Context Protocol (MCP)**: Para integración seamless con Azure DevOps
- **Claude (Anthropic)**: Como modelo de lenguaje principal
- **Azure DevOps REST API**: Para acceso a datos del proyecto

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Usuario       │    │   Agente LLM    │    │   Azure DevOps  │
│                 │───▶│   (LangGraph)   │───▶│      API        │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐
                       │   MCP Server    │
                       │   (ADO Tools)   │
                       └─────────────────┘
```

## 🚀 Características Principales

### Análisis Inteligente
- **Work Items**: Análisis de user stories, bugs, tasks y su progreso
- **Pull Requests**: Revisión de estado, conflictos y métricas de calidad
- **Builds & Releases**: Monitoreo de pipelines y deployments
- **Métricas de Equipo**: Análisis de productividad y carga de trabajo

### Reportes Automatizados
- Resúmenes de sprint en tiempo real
- Identificación de bloqueos y riesgos
- Análisis de tendencias y patrones
- Reportes ejecutivos personalizados

### Asistencia Contextual
- Respuestas inteligentes sobre el estado del proyecto
- Sugerencias basadas en mejores prácticas de DevOps
- Alertas proactivas sobre issues críticos

## 📁 Estructura del Proyecto

```
ado_mcp_agent/
│
├── data/
│ └── config/
│ ├── init.py
│ └── .gitkeep
│
├── src/ # Código fuente del proyecto
│ ├── agent/
│ │ └── tools/
│ │ ├── init.py
│ │ └── .gitkeep
│ │
│ └── mcp_server/
│ ├── init.py
│ └── main.py
│
├── .env
├── .env.example
├── .gitignore
├── pyproject.toml
└── README.md
```

## 🛠️ Instalación y Configuración

### Prerrequisitos

- Python 3.9+
- Cuenta de Azure DevOps con acceso API
- Personal Access Token (PAT) de Azure DevOps
- API Key de Anthropic Claude

### Instalación

1. **Clonar el repositorio**
   ```bash
   git clone <repository-url>
   cd ado_mcp_agent
   ```

2. **Crear entorno virtual**
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   ```

3. **Instalar dependencias**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configurar variables de entorno**
   ```bash
   cp .env.example .env
   ```

   Editar `.env` con tus credenciales:
   ```env
   # Azure DevOps Configuration
   ADO_ORGANIZATION_URL=https://dev.azure.com/your-org
   ADO_PAT_TOKEN=your-personal-access-token
   ADO_PROJECT_NAME=your-project-name

   # Claude API Configuration
   ANTHROPIC_API_KEY=your-anthropic-api-key

   # Agent Configuration
   AGENT_NAME=ADO Assistant
   AGENT_DESCRIPTION=Specialized Azure DevOps AI Assistant
   ```

## 🎮 Uso

### Ejecutar el Agente

```bash
python main.py
```

### Ejemplos de Consultas

```python
# Análisis de Sprint
"¿Cuál es el estado actual del sprint? Identifica posibles riesgos."

# Revisión de Pull Requests
"Dame un resumen de los PRs pendientes y cuáles necesitan atención urgente."

# Métricas de Equipo
"Analiza la carga de trabajo del equipo y sugiere optimizaciones."

# Reportes Ejecutivos
"Genera un reporte semanal del progreso del proyecto para stakeholders."
```

## 🧪 Casos de Uso del POC

### 1. Análisis de Sprint
- Progreso de work items por estado
- Identificación de bloqueos
- Predicción de cumplimiento de objetivos

### 2. Gestión de Pull Requests
- Revisión de PRs pendientes
- Análisis de conflictos y complejidad
- Métricas de tiempo de revisión

### 3. Monitoreo de Builds
- Estado de pipelines CI/CD
- Análisis de fallos recurrentes
- Métricas de tiempo de build

### 4. Reportes Ejecutivos
- KPIs de desarrollo
- Tendencias de productividad
- Análisis de riesgos del proyecto

## 🔧 Tecnologías Utilizadas

- **[LangChain](https://langchain.com/)**: Framework para aplicaciones LLM
- **[LangGraph](https://langchain-ai.github.io/langgraph/)**: Construcción de agentes con grafos de estado
- **[Model Context Protocol](https://modelcontextprotocol.io/)**: Protocolo para integración de contexto
- **[Azure DevOps REST API](https://docs.microsoft.com/en-us/rest/api/azure/devops/)**: API para acceso a datos ADO
- **[Claude (Anthropic)](https://www.anthropic.com/)**: Modelo de lenguaje principal
- **[Pydantic](https://pydantic.dev/)**: Validación de datos y configuración

## 📝 Roadmap

### Fase 1 - POC Base ✅
- [x] Configuración básica del proyecto
- [ ] Implementación del servidor MCP
- [ ] Integración con Azure DevOps API
- [ ] Agente básico con LangGraph

### Fase 2 - Características Avanzadas
- [ ] Análisis predictivo con ML
- [ ] Integración con Teams/Slack
- [ ] Dashboard web interactivo
- [ ] Alertas automáticas

### Fase 3 - Escalabilidad
- [ ] Soporte multi-proyecto
- [ ] Persistencia de contexto
- [ ] API REST para integración
- [ ] Métricas y observabilidad

## 🤝 Contribución

1. Fork el proyecto
2. Crea una rama feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para detalles.

## 📞 Contacto

**Desarrollador**: Fernando
**Email**: [fer.gomez.dev@gmail.com]
**Proyecto**: []

---

> **Nota**: Este es un proyecto de Proof of Concept (POC) diseñado para explorar las capacidades de integración entre LLM y Azure DevOps. No está diseñado para uso en producción sin las debidas consideraciones de seguridad y escalabilidad.