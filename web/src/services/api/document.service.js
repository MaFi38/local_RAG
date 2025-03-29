import axios from "axios";

const API_URL = "http://localhost:5173";

export const uploadDocument = async (file, userId) => {
  const user = localStorage.getItem("user");
  const userp = JSON.parse(user);
  const token = userp.access_token;

  const formData = new FormData();
  formData.append('file', file);
  formData.append('user_id', userp.user_id);

  console.log(userp.user_id);
  try {
    const response = await axios.post(
      `${API_URL}/upload-pdf`,
      formData,
      {
        headers: {
          "Content-Type": "multipart/form-data",
          Authorization: `Bearer ${token}`,
        },
      }
    );
    console.log(response);
    await listDocuments(userp.user_id);
    return response.data;
  } catch (error) {
    console.error("Upload error:", error);
    throw new Error(error.message || "Failed to upload document");
  }
};

export const listDocuments = async (userId) => {
  const user = localStorage.getItem("user");
  const userp = JSON.parse(user);
  const token = userp.access_token;
  
  try {
    const response = await axios.get(`${API_URL}/pdfs/${userp.user_id}`, {
      headers: {
        Authorization: `Bearer ${token}`,
      },
    });
    console.log(response);
    return response.data;
  } catch (error) {
    console.error("List documents error:", error);
    throw new Error(error.response?.data?.detail || "Failed to list documents");
  }
};

export const deleteDocument = async (documentId, userId) => {
  const user = localStorage.getItem("user");
  const userp = JSON.parse(user);
  const token = userp.access_token;
  
  try {
    const response = await axios.post(
      `${API_URL}/delete-doc`,
      {
        file_id: documentId,
        user_id: userp.user_id,
      },
      {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      }
    );
    return response;
  } catch (error) {
    console.error("Delete document error:", error);
    throw new Error(error.response?.data?.detail || "Failed to delete document");
  }
};
