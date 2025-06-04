import React, { useState } from 'react';

type FeedbackProps = {
  responseId: string; // unique ID for the chat response this feedback is for
  onSubmit: (data: {
    responseId: string;
    vote: 'like' | 'dislike' | null;
    comment: string;
  }) => void;
};

const Feedback: React.FC<FeedbackProps> = ({ responseId, onSubmit }) => {
  const [vote, setVote] = useState<'like' | 'dislike' | null>(null);
  const [comment, setComment] = useState('');
  const [submitted, setSubmitted] = useState(false);

  const handleSubmit = () => {
    if (!vote && comment.trim() === '') return; // Do not submit empty feedback
    onSubmit({ responseId, vote, comment });
    setSubmitted(true);
  };

  if (submitted) {
    return <p className="text-green-600 text-sm mt-2">Thank you for your feedback!</p>;
  }

  return (
    <div className="border-t pt-2 mt-4">
      <div className="flex items-center space-x-2 mb-2">
        <button
          className={`px-2 py-1 border rounded ${
            vote === 'like' ? 'bg-green-200 border-green-400' : 'hover:bg-gray-100'
          }`}
          onClick={() => setVote(vote === 'like' ? null : 'like')}
        >
          👍 Like
        </button>
        <button
          className={`px-2 py-1 border rounded ${
            vote === 'dislike' ? 'bg-red-200 border-red-400' : 'hover:bg-gray-100'
          }`}
          onClick={() => setVote(vote === 'dislike' ? null : 'dislike')}
        >
          👎 Dislike
        </button>
      </div>
      <textarea
        className="w-full border p-2 rounded text-sm"
        placeholder="What should the ideal answer be?"
        rows={3}
        value={comment}
        onChange={(e) => setComment(e.target.value)}
      />
      <button
        className="mt-2 px-4 py-1 bg-blue-500 text-white rounded disabled:opacity-50"
        onClick={handleSubmit}
        disabled={!vote && comment.trim() === ''}
      >
        Submit Feedback
      </button>
    </div>
  );
};

export default Feedback;
