import api from './config';

export const userService = {
  async login(email, password) {
    try {
      const response = await api.post('/api/login', {
        email,
        password
      });
      if (response.data.token) {
        localStorage.setItem('token', response.data.token);
        localStorage.setItem('user', JSON.stringify(response.data.user));
      }
      return response.data;
    } catch (error) {
      throw new Error(error.response?.data?.message || 'Failed to login');
    }
  },

  async register(email, password, name) {
    try {
      const response = await api.post('/api/register', {
        email,
        password,
        name
      });
      return response.data;
    } catch (error) {
      throw new Error(error.response?.data?.message || 'Failed to register');
    }
  },

  async logout() {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
  },

  async updateProfile(userId, data) {
    try {
      const response = await api.put(`/api/users/${userId}`, data);
      return response.data;
    } catch (error) {
      throw new Error(error.response?.data?.message || 'Failed to update profile');
    }
  },

  async getPreferences(userId) {
    try {
      const response = await api.get(`/api/users/${userId}/preferences`);
      return response.data;
    } catch (error) {
      throw new Error(error.response?.data?.message || 'Failed to get preferences');
    }
  },

  async updatePreferences(userId, preferences) {
    try {
      const response = await api.put(`/api/users/${userId}/preferences`, preferences);
      return response.data;
    } catch (error) {
      throw new Error(error.response?.data?.message || 'Failed to update preferences');
    }
  }
}; 