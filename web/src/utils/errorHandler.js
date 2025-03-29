export class APIError extends Error {
  constructor(message, status) {
    super(message);
    this.status = status;
    this.name = 'APIError';
  }
}

export const handleAPIError = (error) => {
  if (error.response) {
    throw new APIError(
      error.response.data.message || 'An error occurred',
      error.response.status
    );
  }
  throw new APIError('Network error occurred', 500);
}; 