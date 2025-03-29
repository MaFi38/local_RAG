import { v4 as uuidv4 } from 'uuid';

export const createNewSession = () => {
  return {
    id: uuidv4(),
    messages: [],
    model: "gpt-4",
    timestamp: new Date().toISOString()
  };
};

export const formatTimestamp = (timestamp) => {
  return new Date(timestamp).toLocaleString();
};

export const truncateText = (text, maxLength = 30) => {
  if (text.length <= maxLength) return text;
  return text.substring(0, maxLength) + '...';
}; 