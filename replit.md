# autoPicklr Trading Simulator

## Overview

autoPicklr is a cryptocurrency trading simulator that implements an automated trading strategy using momentum-based signals. The system fetches real-time market data for major cryptocurrencies (BTC, ETH, SOL) from CoinGecko, analyzes price movements using technical indicators (EMA crossovers and breakout patterns), and simulates trading with realistic fees and slippage. The application includes a web dashboard for monitoring portfolio performance, open positions, and trading history in real-time.

## User Preferences

Preferred communication style: Simple, everyday language.

## Recent Changes (August 17, 2025)

✓ Fixed FastAPI deprecated `on_event` decorators - migrated to new `lifespan` context manager
✓ Resolved SQLModel query syntax issues with `.desc()` and `.asc()` methods
✓ Fixed null pointer access issues with wallet balance checks
✓ Corrected SQL execution methods for database operations
✓ Updated all database query ordering to use proper SQLModel syntax
✓ Added proper error handling for wallet operations in trading loop
✓ Created `run_server.py` for easier server startup

## How to Run the Application

1. **Start the server**: Run `python run_server.py` or `uvicorn main:app --host 0.0.0.0 --port 5000 --reload`
2. **Access the dashboard**: Open `http://localhost:5000` in your browser
3. **API endpoints**: Available at `/api/*` routes for data access
4. **Admin controls**: Use `/admin/*` endpoints to control the trading simulation

## System Architecture

### Backend Framework
The application uses FastAPI as the web framework, providing both API endpoints and serving static files. The choice of FastAPI enables async operations for external API calls while maintaining good performance for the web dashboard.

### Data Model & Persistence
The system uses SQLModel with SQLite for data persistence, creating a unified approach to database schema definition and validation. The data model includes:
- **Wallet**: Tracks portfolio balance and equity
- **Candle**: Stores OHLC price data with minute-level granularity
- **Signal**: Records trading signals with entry/exit levels and reasoning
- **Order**: Manages buy/sell orders with execution status
- **Position**: Tracks open positions with stop-loss and take-profit levels
- **Trade**: Records completed trades with P&L calculations

### Trading Engine Architecture
The system follows a modular design with separated concerns:
- **Data Module**: Handles market data fetching and storage
- **Signal Engine**: Implements momentum-based trading strategy using EMA crossovers and breakout detection
- **Risk Management**: Controls position sizing based on percentage risk per trade and maximum open positions
- **Simulation Engine**: Simulates order execution with realistic fees and slippage

### Signal Generation Strategy
The trading strategy combines multiple technical indicators:
- Dual EMA crossover (12/26 periods) for momentum detection
- Breakout analysis using 20-period price highs
- Volume filtering to ensure liquidity
- Composite scoring system with configurable threshold for signal generation

### Configuration Management
Environment-based configuration system allows for easy parameter tuning:
- Trading universe selection
- Risk parameters (position sizing, max positions)
- Technical indicator settings
- Market simulation parameters (fees, slippage)

### Web Interface
Browser-based dashboard using Bootstrap for responsive design, featuring:
- Real-time portfolio metrics display
- Interactive equity curve chart using Chart.js
- Position and trade history tables
- Auto-refreshing data updates

## External Dependencies

### Market Data Provider
- **CoinGecko API**: Primary data source for cryptocurrency prices and volume data
- Provides minute-level OHLCV data for major cryptocurrencies
- Rate-limited to minimize API load while maintaining data freshness

### Python Libraries
- **FastAPI**: Web framework for API and static file serving
- **SQLModel**: Database ORM with Pydantic integration for data validation
- **httpx**: Async HTTP client for external API calls
- **uvicorn**: ASGI server for running the FastAPI application

### Frontend Dependencies
- **Bootstrap**: CSS framework for responsive UI components
- **Chart.js**: JavaScript charting library for portfolio visualization
- **Font Awesome**: Icon library for UI elements

### Database
- **SQLite**: Embedded database for local data persistence
- Chosen for simplicity and ease of deployment without external database requirements
- Suitable for development and small-scale trading simulations