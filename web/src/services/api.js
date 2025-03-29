import * as authService from './auth.service';
import * as chatService from './chat.service';
import * as documentService from './document.service';
import axios from 'axios';

export {
  authService,
  chatService,
  documentService
};

export const setupAxiosInterceptors = (navigate) => {
  axios.interceptors.response.use(
    (response) => response,
    (error) => {
      if (error.response?.status === 401) {
        localStorage.removeItem('user');
        localStorage.removeItem('token');
        navigate('/login');
      }
      return Promise.reject(error);
    }
  );
}; 