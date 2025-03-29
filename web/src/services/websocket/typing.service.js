import { wsService } from './socket.service';

export const typingService = {
  startTyping(sessionId, userId) {
    wsService.socket?.emit('typing:start', {
      sessionId,
      userId
    });
  },

  stopTyping(sessionId, userId) {
    wsService.socket?.emit('typing:stop', {
      sessionId,
      userId
    });
  },

  // Debounced version to prevent too many events
  debouncedTyping: (() => {
    let timeout;
    return (sessionId, userId) => {
      if (timeout) clearTimeout(timeout);
      
      wsService.socket?.emit('typing:start', {
        sessionId,
        userId
      });

      timeout = setTimeout(() => {
        wsService.socket?.emit('typing:stop', {
          sessionId,
          userId
        });
      }, 1000);
    };
  })()
}; 