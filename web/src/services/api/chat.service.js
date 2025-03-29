import axios from 'axios';

const API_URL = 'http://localhost:5173';



export const getChatResponse = async (question, sessionId, model, userId) => {
  const user = localStorage.getItem("user");
  const userp = JSON.parse(user);
  const token = userp.access_token;
  try {
    const response = await axios.post(`${API_URL}/chat`, {
      question,
      session_id: sessionId,
      model,
      user_id: userp.user_id
    }, {
      headers: {
        'Authorization': `Bearer ${token}`
      }
    });
    console.log(response);
    return response.data;
  } catch (error) {
    throw new Error(error.response?.data?.error || 'Failed to get chat response');
  }
};

export const getUserChatHistory = async (userId) => {
  const user = localStorage.getItem("user");
  const userp = JSON.parse(user);
  const token = userp.access_token;
  try {
    const response = await axios.get(`${API_URL}/chat/history/${userId}`, {
      headers: {
        'Authorization': `Bearer ${token}`
      }
    });
    return response.data;
  } catch (error) {
    throw new Error(error.response?.data?.error || 'Failed to get chat history');
  }
}; 