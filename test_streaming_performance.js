/**
 * Performance test script to compare localhost vs ngrok streaming
 * Run this in your browser console while on the chat page
 */

// Configuration - update these URLs based on your setup
const LOCALHOST_URL = 'http://localhost:8000';
const NGROK_URL = 'https://your-ngrok-url.ngrok-free.app'; // Update this!
const TEST_QUESTION = 'What is PTSD?';

// Get auth token from cookies
function getAuthToken() {
  const cookies = document.cookie.split(';');
  for (let cookie of cookies) {
    const [name, value] = cookie.trim().split('=');
    if (name === 'token') {
      return value;
    }
  }
  return null;
}

// Test streaming performance
async function testStreamingPerformance(baseUrl, endpoint, description) {
  const token = getAuthToken();
  if (!token) {
    console.error('❌ No auth token found. Please login first.');
    return null;
  }

  console.log(`🧪 Testing ${description}...`);
  const startTime = performance.now();
  let firstChunkTime = 0;
  let chunksReceived = 0;
  let totalCharsReceived = 0;

  try {
    const response = await fetch(`${baseUrl}${endpoint}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`,
        'ngrok-skip-browser-warning': 'true'
      },
      body: JSON.stringify({
        question: TEST_QUESTION,
        session_id: null
      })
    });

    if (!response.body) {
      throw new Error("Response body is null");
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');

    while (true) {
      const chunkStart = performance.now();
      const { value, done } = await reader.read();
      
      if (done) break;

      const chunk = decoder.decode(value, { stream: true });
      const chunkEnd = performance.now();
      
      chunksReceived++;
      totalCharsReceived += chunk.length;
      
      if (firstChunkTime === 0) {
        firstChunkTime = chunkEnd - startTime;
      }

      // Log first few chunks for debugging
      if (chunksReceived <= 3) {
        console.log(`  📦 Chunk ${chunksReceived}: ${(chunkEnd - chunkStart).toFixed(2)}ms, Length: ${chunk.length}`);
      }
    }

    const totalTime = performance.now() - startTime;
    
    const result = {
      description,
      firstChunkTime: firstChunkTime.toFixed(2),
      totalTime: totalTime.toFixed(2),
      chunksReceived,
      totalCharsReceived,
      avgChunkTime: (totalTime / chunksReceived).toFixed(2),
      throughput: (totalCharsReceived / (totalTime / 1000)).toFixed(2)
    };

    console.log(`✅ ${description} completed:`, result);
    return result;

  } catch (error) {
    console.error(`❌ ${description} failed:`, error);
    return { 
      description, 
      error: error.message,
      firstChunkTime: 0,
      totalTime: (performance.now() - startTime).toFixed(2)
    };
  }
}

// Main comparison function
async function compareStreamingPerformance() {
  console.log('🚀 Starting streaming performance comparison...');
  console.log('================================================');
  
  // Test the optimized endpoint on localhost
  const localhostOptimized = await testStreamingPerformance(
    LOCALHOST_URL, 
    '/stream_async_optimized', 
    'Localhost (Optimized)'
  );
  
  await new Promise(resolve => setTimeout(resolve, 2000)); // Wait 2 seconds
  
  // Test the optimized endpoint via ngrok
  const ngrokOptimized = await testStreamingPerformance(
    NGROK_URL, 
    '/stream_async_optimized', 
    'Ngrok (Optimized)'
  );
  
  await new Promise(resolve => setTimeout(resolve, 2000)); // Wait 2 seconds
  
  // Test regular endpoint on localhost
  const localhostRegular = await testStreamingPerformance(
    LOCALHOST_URL, 
    '/stream_async', 
    'Localhost (Regular)'
  );
  
  await new Promise(resolve => setTimeout(resolve, 2000)); // Wait 2 seconds
  
  // Test regular endpoint via ngrok
  const ngrokRegular = await testStreamingPerformance(
    NGROK_URL, 
    '/stream_async', 
    'Ngrok (Regular)'
  );

  // Generate comparison report
  console.log('\n📊 PERFORMANCE COMPARISON REPORT');
  console.log('=====================================');
  
  if (localhostOptimized && ngrokOptimized) {
    const optimizedLatencyOverhead = parseFloat(ngrokOptimized.firstChunkTime) - parseFloat(localhostOptimized.firstChunkTime);
    console.log(`🔥 Optimized Endpoint:`);
    console.log(`   Localhost first chunk: ${localhostOptimized.firstChunkTime}ms`);
    console.log(`   Ngrok first chunk:     ${ngrokOptimized.firstChunkTime}ms`);
    console.log(`   Ngrok latency overhead: +${optimizedLatencyOverhead.toFixed(2)}ms`);
  }
  
  if (localhostRegular && ngrokRegular) {
    const regularLatencyOverhead = parseFloat(ngrokRegular.firstChunkTime) - parseFloat(localhostRegular.firstChunkTime);
    console.log(`\n⚙️  Regular Endpoint:`);
    console.log(`   Localhost first chunk: ${localhostRegular.firstChunkTime}ms`);
    console.log(`   Ngrok first chunk:     ${ngrokRegular.firstChunkTime}ms`);
    console.log(`   Ngrok latency overhead: +${regularLatencyOverhead.toFixed(2)}ms`);
  }
  
  if (localhostOptimized && localhostRegular) {
    const optimizationGain = parseFloat(localhostRegular.firstChunkTime) - parseFloat(localhostOptimized.firstChunkTime);
    console.log(`\n💡 Optimization Impact (Localhost):`);
    console.log(`   Regular:   ${localhostRegular.firstChunkTime}ms`);
    console.log(`   Optimized: ${localhostOptimized.firstChunkTime}ms`);
    console.log(`   Improvement: -${optimizationGain.toFixed(2)}ms (${((optimizationGain/parseFloat(localhostRegular.firstChunkTime))*100).toFixed(1)}%)`);
  }
  
  console.log('\n💡 Recommendations:');
  console.log('==================');
  console.log('1. Use /stream_async_optimized for better performance');
  console.log('2. For production, consider using a proper domain instead of ngrok');
  console.log('3. Test locally when possible for development');
  console.log('4. Monitor the frontend optimizations implemented');
}

// Instructions
console.log('📋 Instructions:');
console.log('================');
console.log('1. Update the NGROK_URL in this script with your actual ngrok URL');
console.log('2. Make sure you are logged into the chat application');
console.log('3. Run: compareStreamingPerformance()');
console.log('');
console.log('🔧 Quick test (optimized endpoint only):');
console.log('testStreamingPerformance("http://localhost:8000", "/stream_async_optimized", "Quick Test")');

// Export the functions
window.testStreamingPerformance = testStreamingPerformance;
window.compareStreamingPerformance = compareStreamingPerformance;
