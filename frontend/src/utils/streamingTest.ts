/**
 * Utility to test streaming performance with different configurations
 * Use this to measure latency differences between ngrok and localhost
 */

interface StreamingTestConfig {
  baseUrl: string;
  endpoint: string;
  question: string;
  token?: string;
  sessionId?: string;
}

interface StreamingTestResult {
  timeToFirstChunk: number;
  totalTime: number;
  chunksReceived: number;
  averageChunkLatency: number;
  success: boolean;
  error?: string;
}

export class StreamingPerformanceTester {
  static async testStreamingPerformance(config: StreamingTestConfig): Promise<StreamingTestResult> {
    const startTime = performance.now();
    let firstChunkTime = 0;
    let chunksReceived = 0;
    let lastChunkTime = startTime;
    
    try {
      const headers: HeadersInit = {
        'Content-Type': 'application/json',
        'ngrok-skip-browser-warning': 'true'
      };
      
      if (config.token) {
        headers['Authorization'] = `Bearer ${config.token}`;
      }

      const response = await fetch(`${config.baseUrl}${config.endpoint}`, {
        method: 'POST',
        headers,
        body: JSON.stringify({
          question: config.question,
          session_id: config.sessionId
        })
      });

      if (!response.body) {
        throw new Error("Response body is null");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');

      while (true) {
        const chunkStartTime = performance.now();
        const { value, done } = await reader.read();
        
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        const chunkEndTime = performance.now();
        
        chunksReceived++;
        
        if (firstChunkTime === 0) {
          firstChunkTime = chunkEndTime - startTime;
        }
        
        lastChunkTime = chunkEndTime;
        
        // Log chunk info for debugging
        console.log(`Chunk ${chunksReceived}: ${(chunkEndTime - chunkStartTime).toFixed(2)}ms, Content: ${chunk.slice(0, 50)}...`);
      }

      const totalTime = lastChunkTime - startTime;
      const averageChunkLatency = totalTime / chunksReceived;

      return {
        timeToFirstChunk: firstChunkTime,
        totalTime,
        chunksReceived,
        averageChunkLatency,
        success: true
      };

    } catch (error) {
      return {
        timeToFirstChunk: 0,
        totalTime: performance.now() - startTime,
        chunksReceived,
        averageChunkLatency: 0,
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error'
      };
    }
  }

  /**
   * Compare performance between localhost and ngrok
   */
  static async comparePerformance(
    localhostUrl: string,
    ngrokUrl: string,
    endpoint: string,
    question: string,
    token?: string,
    sessionId?: string
  ): Promise<{ localhost: StreamingTestResult; ngrok: StreamingTestResult }> {
    
    console.log('🚀 Starting performance comparison...');
    
    // Test localhost
    console.log('📡 Testing localhost...');
    const localhostResult = await this.testStreamingPerformance({
      baseUrl: localhostUrl,
      endpoint,
      question,
      token,
      sessionId
    });

    // Wait a bit between tests
    await new Promise(resolve => setTimeout(resolve, 1000));

    // Test ngrok
    console.log('🌐 Testing ngrok...');
    const ngrokResult = await this.testStreamingPerformance({
      baseUrl: ngrokUrl,
      endpoint,
      question,
      token,
      sessionId
    });

    // Log comparison
    console.log('\n📊 Performance Comparison Results:');
    console.log('======================================');
    console.log(`Localhost - First chunk: ${localhostResult.timeToFirstChunk.toFixed(2)}ms, Total: ${localhostResult.totalTime.toFixed(2)}ms`);
    console.log(`Ngrok     - First chunk: ${ngrokResult.timeToFirstChunk.toFixed(2)}ms, Total: ${ngrokResult.totalTime.toFixed(2)}ms`);
    console.log(`Latency overhead: ${(ngrokResult.timeToFirstChunk - localhostResult.timeToFirstChunk).toFixed(2)}ms`);
    console.log('======================================\n');

    return {
      localhost: localhostResult,
      ngrok: ngrokResult
    };
  }
}

// Export convenience function for easy testing in browser console
export const testStreaming = StreamingPerformanceTester.testStreamingPerformance;
export const compareStreaming = StreamingPerformanceTester.comparePerformance;
