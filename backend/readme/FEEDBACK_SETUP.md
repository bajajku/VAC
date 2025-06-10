# Feedback System Setup Guide

This guide will help you set up the MongoDB-based feedback system for the VAC AI Assistant.

## Prerequisites

- Python 3.8+
- Node.js 16+
- MongoDB (local or cloud)

## 1. MongoDB Setup

### Option A: Local MongoDB Installation

#### macOS (using Homebrew)
```bash
# Install MongoDB
brew tap mongodb/brew
brew install mongodb-community

# Start MongoDB service
brew services start mongodb/brew/mongodb-community

# Verify installation
mongosh --eval "db.runCommand({ connectionStatus: 1 })"
```

#### Ubuntu/Debian
```bash
# Import MongoDB public key
wget -qO - https://www.mongodb.org/static/pgp/server-7.0.asc | sudo apt-key add -

# Add MongoDB repository
echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu focal/mongodb-org/7.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-7.0.list

# Install MongoDB
sudo apt-get update
sudo apt-get install -y mongodb-org

# Start MongoDB
sudo systemctl start mongod
sudo systemctl enable mongod
```

#### Windows
1. Download MongoDB Community Server from https://www.mongodb.com/download-center/community
2. Run the installer and follow the setup wizard
3. Start MongoDB service from Services panel or command line

### Option B: MongoDB Atlas (Cloud)
1. Go to https://www.mongodb.com/atlas
2. Create a free account and cluster
3. Get your connection string (format: `mongodb+srv://username:password@cluster.mongodb.net/`)

## 2. Backend Setup

### Install Dependencies
```bash
cd backend
pip install motor pymongo
# or if using the requirements.txt
pip install -r requirements.txt
```

### Environment Configuration
1. Copy the environment template:
```bash
cp env.example .env
```

2. Update `.env` with your MongoDB configuration:
```env
# MongoDB Configuration
MONGODB_URL=mongodb://localhost:27017  # For local MongoDB
# OR for MongoDB Atlas:
# MONGODB_URL=mongodb+srv://username:password@cluster.mongodb.net/
MONGODB_DATABASE=vac_feedback

# Other existing configurations...
TOGETHER_API_KEY=your_together_api_key_here
```

### Start the Backend
```bash
cd backend
python -m uvicorn api:app_api --reload --host 0.0.0.0 --port 8000
```

The backend will automatically:
- Connect to MongoDB on startup
- Create the feedback collection if it doesn't exist
- Display connection status in the logs

## 3. Frontend Setup

### Install Dependencies
```bash
cd frontend
npm install
```

### Environment Configuration
Create or update `.env.local`:
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### Start the Frontend
```bash
cd frontend
npm run dev
```

## 4. Testing the Feedback System

### Manual Testing
1. Open http://localhost:3000 in your browser
2. Go to the Chat page
3. Ask a question and get a response
4. Use the feedback buttons (Yes/No) or "More feedback" option
5. Go to the Feedback Dashboard to view submitted feedback

### API Testing with curl
```bash
# Create feedback
curl -X POST "http://localhost:8000/feedback" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "test-session-123",
    "question": "What is AI?",
    "answer": "AI is artificial intelligence...",
    "feedback_type": "positive",
    "rating": 5,
    "feedback_text": "Great explanation!"
  }'

# Get feedback stats
curl "http://localhost:8000/feedback-stats"

# Get session feedback
curl "http://localhost:8000/feedback/session/test-session-123"
```

## 5. Available API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/feedback` | Create new feedback |
| GET | `/feedback/{feedback_id}` | Get feedback by ID |
| GET | `/feedback/session/{session_id}` | Get all feedback for a session |
| PUT | `/feedback/{feedback_id}` | Update feedback |
| DELETE | `/feedback/{feedback_id}` | Delete feedback |
| GET | `/feedback-stats` | Get feedback statistics |

## 6. Database Schema

The feedback collection stores documents with this structure:
```json
{
  "_id": "ObjectId",
  "session_id": "string",
  "question": "string", 
  "answer": "string",
  "feedback_type": "positive|negative|suggestion",
  "feedback_text": "string (optional)",
  "rating": "number 1-5 (optional)",
  "user_id": "string (optional)",
  "metadata": "object (optional)",
  "created_at": "datetime",
  "updated_at": "datetime"
}
```

## 7. Troubleshooting

### MongoDB Connection Issues
- **Local MongoDB**: Ensure MongoDB service is running (`brew services start mongodb-community` on macOS)
- **Atlas**: Check your connection string and network access settings
- **Firewall**: Ensure port 27017 (local) or 27017+ (Atlas) is accessible

### Backend Issues
- Check logs for MongoDB connection status
- Verify environment variables are set correctly
- Ensure required dependencies are installed

### Frontend Issues
- Verify `NEXT_PUBLIC_API_URL` points to the correct backend URL
- Check browser console for API errors
- Ensure backend is running and accessible

## 8. Production Considerations

### Security
- Use MongoDB authentication in production
- Configure network security (firewall, VPN)
- Use environment variables for sensitive data
- Enable HTTPS for API endpoints

### Performance
- Add database indexes for frequent queries:
  ```javascript
  db.feedback.createIndex({ "session_id": 1 })
  db.feedback.createIndex({ "created_at": -1 })
  db.feedback.createIndex({ "feedback_type": 1 })
  ```

### Monitoring
- Set up MongoDB monitoring (Atlas has built-in monitoring)
- Monitor API response times and error rates
- Set up logging for production debugging

## 9. Development Tips

### MongoDB GUI Tools
- **MongoDB Compass**: Official GUI tool
- **Studio 3T**: Advanced MongoDB IDE
- **Robo 3T**: Lightweight MongoDB GUI

### Useful MongoDB Commands
```javascript
// View feedback collection
db.feedback.find().limit(10)

// Count feedback by type
db.feedback.aggregate([
  { $group: { _id: "$feedback_type", count: { $sum: 1 } } }
])

// Get recent feedback
db.feedback.find().sort({ created_at: -1 }).limit(5)

// Delete test data
db.feedback.deleteMany({ session_id: /test/ })
```

This setup provides a complete feedback system with MongoDB persistence, RESTful API endpoints, and a React-based dashboard for viewing feedback analytics. 