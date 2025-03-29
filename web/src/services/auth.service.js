import axios from 'axios';

const API_URL = 'http://localhost:5173';

export const loginUser = async (email, password) => {
  try {
    const response = await axios.post(`${API_URL}/login`, {
      email,
      password,
    });
    console.log(response);
    return response.data;
  } catch (error) {
    console.log(error);
    throw new Error(error.response?.data?.error || 'Login failed');
  }
};

export const registerUser = async (email, password) => {
  try {
    const response = await axios.post(`${API_URL}/register`, {
      email,
      password,
    });
    return response.data;
  } catch (error) {
    throw new Error(error.response?.data?.error || 'Registration failed');
  }
};

export const resetPassword = async (email, newPassword) => {
  try {
    const response = await axios.post(`${API_URL}/reset`, {
      email,
      password: newPassword,
    });
    return response.data;
  } catch (error) {
    throw new Error(error.response?.data?.error || 'Password reset failed');
  }
}; 