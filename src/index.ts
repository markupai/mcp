#!/usr/bin/env node
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
  Tool,
} from '@modelcontextprotocol/sdk/types.js';
import FormData from 'form-data';
import fetch from 'node-fetch';
import { fileURLToPath } from 'url';
import { dirname } from 'path';
import dotenv from 'dotenv';

// Load environment variables
dotenv.config();

const __filename = fileURLToPath(import.meta.url);
dirname(__filename); // Required for ES modules but not used directly

// Logging configuration
const DEBUG = process.env.DEBUG === 'true';
// eslint-disable-next-line @typescript-eslint/no-unused-vars
const LOG_LEVELS = ['error', 'warn', 'info', 'debug'] as const;
type LogLevel = typeof LOG_LEVELS[number];

function log(level: LogLevel, message: string, data?: unknown) {
  if (level === 'debug' && !DEBUG) return;
  const timestamp = new Date().toISOString();
  const logData = data ? ` ${JSON.stringify(data)}` : '';
  console.error(`[${timestamp}] [${level.toUpperCase()}] ${message}${logData}`);
}

// Validate required environment variables
const REQUIRED_ENV_VARS = ['MARKUPAI_API_KEY'];
const missingVars = REQUIRED_ENV_VARS.filter(varName => !process.env[varName]);
if (missingVars.length > 0) {
  log('error', 'Missing required environment variables', missingVars);
  console.error('\nPlease set the following environment variables:');
  missingVars.forEach(varName => console.error(`  - ${varName}`));
  console.error('\nYou can create a .env file with these variables.');
  process.exit(1);
}

// Configuration
const MARKUPAI_BASE_URL = process.env.MARKUPAI_BASE_URL || 'https://api.markup.ai';
const MARKUPAI_API_KEY = process.env.MARKUPAI_API_KEY!;
const WORKFLOW_TIMEOUT = parseInt(process.env.WORKFLOW_TIMEOUT || '60000', 10);
const POLL_INTERVAL = parseInt(process.env.POLL_INTERVAL || '2000', 10);
const MAX_RETRIES = parseInt(process.env.MAX_RETRIES || '3', 10);

log('info', 'Configuration loaded', {
  baseUrl: MARKUPAI_BASE_URL,
  workflowTimeout: WORKFLOW_TIMEOUT,
  pollInterval: POLL_INTERVAL,
  maxRetries: MAX_RETRIES
});

// Style guide name to UUID mapping
const STYLE_GUIDE_IDS: Record<string, string> = {
  'ap': '01971e03-dd27-75ee-9044-b48e654848cf',
  'chicago': '01971e03-dd27-77d8-a6fa-5edb6a1f4ad2',
  'microsoft': '01971e03-dd27-779f-b3ec-b724a2cf809f',
  'proofpoint': '01971e03-dd27-7dfa-8d96-d48c8cf5e4fe',
};

// Type definitions
interface BaseToolArgs {
  text: string;
  dialect?: string;
  tone?: string;
  style_guide?: string;
}

interface WorkflowStatusArgs {
  workflow_id: string;
  workflow_type: 'rewrites' | 'checks' | 'suggestions';
}


interface Position {
  start_index?: number;
  end_index?: number;
}

interface Issue {
  original: string;
  position?: Position;
  category?: string;
  subcategory?: string;
}

interface Suggestion {
  original: string;
  suggestion?: string;
  modified?: string;
  position?: Position;
  category?: string;
  subcategory?: string;
}

interface WorkflowInfo {
  id: string;
  api_version?: string;
  generated_at?: string | null;
  status: 'running' | 'completed' | 'failed';
}

interface StyleGuideConfig {
  style_guide_type?: string;
  style_guide_id?: string;
}

interface ConfigOptions {
  dialect?: string;
  style_guide?: StyleGuideConfig | string;
  tone?: string;
}

interface ContentScoreData {
  quality_score?: number;
  clarity_score?: number;
  word_count?: number;
  sentence_count?: number;
  avg_sentence_length?: number;
  flesch_reading_ease?: number;
  flesch_kincaid_grade?: number;
}

interface IssueScoreData {
  grammar_score?: number;
  grammar_issues?: number;
  style_guide_score?: number;
  style_guide_issues?: number;
  terminology_score?: number;
  terminology_issues?: number;
  tone_score?: number;
  informality?: number;
  target_informality?: number;
  liveliness?: number;
  target_liveliness?: number;
}

interface ContentScores {
  content_score?: ContentScoreData;
  issue_score?: IssueScoreData;
}

interface OriginalContent {
  issues?: Issue[];
  initial_scores?: ContentScores;
  final_scores?: ContentScores;
}

interface RewriteContent {
  output?: {
    merged_text?: string;
    original_text?: string;
    initial_scores?: ContentScores;
    final_scores?: ContentScores;
  };
}

interface SuggestionOriginalContent {
  issues?: Suggestion[];
  initial_scores?: ContentScores;
  final_scores?: ContentScores;
}

interface StyleCheckResponse {
  workflow: WorkflowInfo;
  config?: ConfigOptions | null;
  original?: OriginalContent | null;
}

interface RewriteResponse {
  workflow: WorkflowInfo;
  config?: ConfigOptions | null;
  original?: SuggestionOriginalContent | null;
  rewrite?: RewriteContent | null;
}

interface SuggestionResponse {
  workflow: WorkflowInfo;
  config?: ConfigOptions | null;
  original?: SuggestionOriginalContent | null;
}

interface WorkflowSubmitResponse {
  status: 'running' | 'completed' | 'failed';
  workflow_id: string;
}

// Type guards
function isBaseToolArgs(args: unknown): args is BaseToolArgs {
  return (
    typeof args === 'object' &&
    args !== null &&
    'text' in args &&
    typeof (args as Record<string, unknown>).text === 'string'
  );
}

function isWorkflowStatusArgs(args: unknown): args is WorkflowStatusArgs {
  return (
    typeof args === 'object' &&
    args !== null &&
    'workflow_id' in args &&
    'workflow_type' in args &&
    typeof (args as Record<string, unknown>).workflow_id === 'string' &&
    ['rewrites', 'checks', 'suggestions'].includes((args as Record<string, unknown>).workflow_type as string)
  );
}

// Input validation
function validateTextInput(text: string): void {
  if (!text || text.trim().length === 0) {
    throw new Error('Text parameter is required and must be a non-empty string');
  }

  const MAX_TEXT_LENGTH = parseInt(process.env.MAX_TEXT_LENGTH || '100000', 10);
  if (text.length > MAX_TEXT_LENGTH) {
    throw new Error(`Text exceeds maximum length of ${MAX_TEXT_LENGTH} characters`);
  }
}

// Retry logic with exponential backoff
async function retryWithBackoff<T>(
  fn: () => Promise<T>,
  operationName: string,
  maxRetries = MAX_RETRIES,
  baseDelay = 1000
): Promise<T> {
  for (let i = 0; i < maxRetries; i++) {
    try {
      log('debug', `Attempting ${operationName} (attempt ${i + 1}/${maxRetries})`);
      const result = await fn();
      if (i > 0) {
        log('info', `${operationName} succeeded after ${i + 1} attempts`);
      }
      return result;
    } catch (error) {
      const isLastAttempt = i === maxRetries - 1;
      const delay = baseDelay * Math.pow(2, i);

      log('warn', `${operationName} failed (attempt ${i + 1}/${maxRetries})`, {
        error: error instanceof Error ? error.message : error,
        willRetry: !isLastAttempt,
        nextDelayMs: !isLastAttempt ? delay : undefined
      });

      if (isLastAttempt) {
        throw new Error(`${operationName} failed after ${maxRetries} attempts: ${error instanceof Error ? error.message : error}`);
      }

      await new Promise(resolve => setTimeout(resolve, delay));
    }
  }
  throw new Error(`${operationName}: Max retries exceeded`);
}

// Define available tools
const TOOLS: Tool[] = [
  {
    name: 'markupai_rewrite',
    description: 'Automatically rewrite and improve text content using AI-powered style guides. This tool analyzes your text for grammar, clarity, tone, and style guide compliance, then provides a completely rewritten version. Use this when you need to transform rough drafts into polished content, ensure consistency with brand voice, or adapt content for different audiences. Returns both before/after scores and the rewritten text.',
    inputSchema: {
      type: 'object',
      properties: {
        text: {
          type: 'string',
          description: 'The text content to rewrite',
          minLength: 1,
        },
        dialect: {
          type: 'string',
          description: 'Language dialect',
          enum: ['american_english', 'british_oxford', 'canadian_english'],
          default: 'american_english'
        },
        tone: {
          type: 'string',
          description: 'Desired tone of the rewritten content',
          enum: ['academic', 'confident', 'conversational', 'empathetic', 'engaging', 'friendly', 'professional', 'technical'],
          default: 'professional'
        },
        style_guide: {
          type: 'string',
          description: 'Style guide to follow (predefined: ap, chicago, microsoft, proofpoint) or custom UUID',
          default: 'microsoft'
        }
      },
      required: ['text']
    }
  },
  {
    name: 'markupai_check',
    description: 'Analyze text for quality issues without making changes. This tool provides detailed scores for grammar, clarity, tone, style guide compliance, and terminology. Use this for content audits, quality assessments, or when you want to understand specific issues before editing. Returns comprehensive readability metrics and issue counts by category.',
    inputSchema: {
      type: 'object',
      properties: {
        text: {
          type: 'string',
          description: 'The text content to analyze',
          minLength: 1,
        },
        dialect: {
          type: 'string',
          description: 'Language dialect',
          enum: ['american_english', 'british_oxford', 'canadian_english'],
          default: 'american_english'
        },
        tone: {
          type: 'string',
          description: 'Target tone to check against',
          enum: ['academic', 'confident', 'conversational', 'empathetic', 'engaging', 'friendly', 'professional', 'technical'],
          default: 'professional'
        },
        style_guide: {
          type: 'string',
          description: 'Style guide to check against (predefined: ap, chicago, microsoft) or custom UUID',
          default: 'microsoft'
        }
      },
      required: ['text']
    }
  },
  {
    name: 'markupai_suggestions',
    description: 'Get detailed editing suggestions for improving text. This tool identifies specific issues and provides targeted recommendations for each problem found. Use this when you want to maintain editorial control while getting guidance on improvements. Returns a categorized list of issues with specific suggestions for each.',
    inputSchema: {
      type: 'object',
      properties: {
        text: {
          type: 'string',
          description: 'The text content to get suggestions for',
          minLength: 1,
        },
        dialect: {
          type: 'string',
          description: 'Language dialect',
          enum: ['american_english', 'british_oxford', 'canadian_english'],
          default: 'american_english'
        },
        tone: {
          type: 'string',
          description: 'Target tone for suggestions',
          enum: ['academic', 'confident', 'conversational', 'empathetic', 'engaging', 'friendly', 'professional', 'technical'],
          default: 'professional'
        },
        style_guide: {
          type: 'string',
          description: 'Style guide for suggestions (predefined: ap, chicago, microsoft) or custom UUID',
          default: 'microsoft'
        }
      },
      required: ['text']
    }
  },
  {
    name: 'markupai_workflow_status',
    description: 'Check the status of an asynchronous Markup AI workflow. Use this to poll for results when other operations return a running status. Workflows typically complete within 5-30 seconds depending on text length and complexity.',
    inputSchema: {
      type: 'object',
      properties: {
        workflow_id: {
          type: 'string',
          description: 'The workflow ID to check status for'
        },
        workflow_type: {
          type: 'string',
          description: 'The type of workflow',
          enum: ['rewrites', 'checks', 'suggestions']
        }
      },
      required: ['workflow_id', 'workflow_type']
    }
  }
];

// Create a buffer for upload
function createTextBuffer(text: string): Buffer {
  return Buffer.from(text, 'utf-8');
}

// Submit a workflow to Markup AI
async function submitWorkflow(
  endpoint: 'rewrites' | 'checks' | 'suggestions',
  text: string,
  dialect: string,
  tone: string,
  style_guide: string
): Promise<WorkflowSubmitResponse> {
  validateTextInput(text);

  return retryWithBackoff(async () => {
    const formData = new FormData();
    const textBuffer = createTextBuffer(text);
    formData.append('file_upload', textBuffer, {
      filename: 'content.txt',
      contentType: 'text/plain'
    });
    formData.append('dialect', dialect);
    formData.append('tone', tone);

    // Convert style guide name to UUID
    const styleGuideId = STYLE_GUIDE_IDS[style_guide] || style_guide;
    formData.append('style_guide', styleGuideId);

    log('debug', `Submitting ${endpoint} workflow`, {
      dialect,
      tone,
      style_guide: styleGuideId,
      textLength: text.length
    });

    const response = await fetch(`${MARKUPAI_BASE_URL}/v1/style/${endpoint}`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${MARKUPAI_API_KEY}`,
        ...formData.getHeaders()
      },
      body: formData,
    });

    if (!response.ok) {
      const errorText = await response.text();
      log('error', `API request failed`, {
        status: response.status,
        statusText: response.statusText,
        error: errorText
      });
      throw new Error(`Markup AI API error: ${response.status} ${response.statusText} - ${errorText}`);
    }

    const result = await response.json() as WorkflowSubmitResponse;
    log('info', `Workflow submitted successfully`, {
      workflow_id: result.workflow_id,
      status: result.status
    });

    return result;
  }, `submit ${endpoint} workflow`);
}

// Get workflow status
async function getWorkflowStatus(
  workflow_id: string,
  workflow_type: 'rewrites' | 'checks' | 'suggestions'
): Promise<StyleCheckResponse | RewriteResponse | SuggestionResponse> {
  return retryWithBackoff(async () => {
    log('debug', `Checking workflow status`, { workflow_id, workflow_type });

    const response = await fetch(
      `${MARKUPAI_BASE_URL}/v1/style/${workflow_type}/${workflow_id}`,
      {
        headers: {
          'Authorization': `Bearer ${MARKUPAI_API_KEY}`,
        },
      }
    );

    if (!response.ok) {
      const errorText = await response.text();
      log('error', `Status check failed`, {
        status: response.status,
        statusText: response.statusText,
        error: errorText
      });
      throw new Error(`Markup AI API error: ${response.status} ${response.statusText} - ${errorText}`);
    }

    const result = await response.json() as StyleCheckResponse | RewriteResponse | SuggestionResponse;
    log('debug', `Workflow status retrieved`, {
      workflow_id,
      status: result.workflow.status
    });

    return result;
  }, 'check workflow status');
}

// Poll workflow until completion with timeout
async function pollWorkflowCompletion(
  workflow_id: string,
  workflow_type: 'rewrites' | 'checks' | 'suggestions'
): Promise<StyleCheckResponse | RewriteResponse | SuggestionResponse> {
  const startTime = Date.now();
  let status: string = 'running';

  log('info', `Polling workflow for completion`, {
    workflow_id,
    workflow_type,
    timeout: WORKFLOW_TIMEOUT
  });

  let result: StyleCheckResponse | RewriteResponse | SuggestionResponse | null = null;

  while (status === 'running') {
    const elapsedTime = Date.now() - startTime;

    if (elapsedTime > WORKFLOW_TIMEOUT) {
      log('error', `Workflow timeout`, {
        workflow_id,
        elapsedTime,
        timeout: WORKFLOW_TIMEOUT
      });
      throw new Error(`Workflow timeout after ${WORKFLOW_TIMEOUT}ms. Workflow ID: ${workflow_id}`);
    }

    await new Promise(resolve => setTimeout(resolve, POLL_INTERVAL));

    try {
      result = await getWorkflowStatus(workflow_id, workflow_type);
      status = result.workflow.status;

      log('debug', `Workflow poll update`, {
        workflow_id,
        status,
        elapsedTime
      });
    } catch (error) {
      log('warn', `Error during polling, will retry`, {
        workflow_id,
        error: error instanceof Error ? error.message : error
      });
      // Continue polling even if individual status checks fail
    }
  }

  if (status === 'failed') {
    log('error', `Workflow failed`, {
      workflow_id
    });
    throw new Error(`Workflow failed`);
  }

  if (!result) {
    throw new Error('No result obtained from workflow');
  }

  log('info', `Workflow completed successfully`, {
    workflow_id,
    totalTime: Date.now() - startTime
  });

  return result;
}

// Format response with highlighted scores
function formatResponse(result: StyleCheckResponse | RewriteResponse | SuggestionResponse): string {
  let formatted = '';

  // Workflow info
  if (result.workflow) {
    formatted += `Status: ${result.workflow.status}\n`;
    formatted += `Workflow ID: ${result.workflow.id}\n`;
  }

  // Config info
  if (result.config) {
    formatted += '\n=== CONFIGURATION ===\n';
    if (result.config.dialect) formatted += `Dialect: ${result.config.dialect}\n`;
    if (result.config.tone) formatted += `Tone: ${result.config.tone}\n`;
    if (result.config.style_guide) {
      const styleGuideInfo = typeof result.config.style_guide === 'object'
        ? (result.config.style_guide as StyleGuideConfig).style_guide_type || (result.config.style_guide as StyleGuideConfig).style_guide_id
        : result.config.style_guide;
      formatted += `Style Guide: ${styleGuideInfo}\n`;
    }
  }

  // Original content scores and issues
  if (result.original) {
    if (result.original.initial_scores || result.original.final_scores) {
      formatted += '\n=== ORIGINAL SCORES ===\n';

      const scores = result.original.final_scores || result.original.initial_scores;
      if (scores) {
        formatted += formatContentScores(scores);
      }
    }

    // Format issues
    if (result.original.issues && result.original.issues.length > 0) {
      formatted += `\n=== ISSUES (${result.original.issues.length} total) ===\n`;
      result.original.issues.forEach((issue, idx) => {
        const suggestion = 'suggestion' in issue ? issue.suggestion : undefined;
        const modified = 'modified' in issue ? issue.modified : undefined;
        const category = issue.category || issue.subcategory || 'General';
        const replacement = suggestion || modified || 'N/A';
        formatted += `${idx + 1}. [${category}] "${issue.original}" → "${replacement}"\n`;
      });
    }
  }

  // Rewrite content (for RewriteResponse)
  if ('rewrite' in result && result.rewrite) {
    if (result.rewrite.output) {
      // Rewrite scores
      if (result.rewrite.output.final_scores || result.rewrite.output.initial_scores) {
        formatted += '\n=== REWRITE SCORES ===\n';
        const scores = result.rewrite.output.final_scores || result.rewrite.output.initial_scores;
        if (scores) {
          formatted += formatContentScores(scores);
        }
      }

      // Rewritten text
      if (result.rewrite.output.merged_text) {
        formatted += '\n=== REWRITTEN TEXT ===\n';
        formatted += result.rewrite.output.merged_text + '\n';
      }
    }
  }

  if (DEBUG) {
    formatted += '\n=== FULL RESPONSE ===\n';
    formatted += JSON.stringify(result, null, 2);
  }

  return formatted;
}

// Helper function to format content scores
function formatContentScores(scores: ContentScores): string {
  let formatted = '';

  // Extract and format scores based on the actual structure
  if (scores.content_score) {
    const cs = scores.content_score;
    if (cs.quality_score !== undefined) {
      formatted += `Quality Score: ${cs.quality_score}\n`;
    }
    if (cs.clarity_score !== undefined) {
      formatted += `Clarity Score: ${cs.clarity_score}\n`;
    }
    if (cs.word_count !== undefined) {
      formatted += `  - Word Count: ${cs.word_count}\n`;
    }
    if (cs.sentence_count !== undefined) {
      formatted += `  - Sentence Count: ${cs.sentence_count}\n`;
    }
    if (cs.avg_sentence_length !== undefined) {
      formatted += `  - Avg Sentence Length: ${cs.avg_sentence_length}\n`;
    }
    if (cs.flesch_reading_ease !== undefined) {
      formatted += `  - Flesch Reading Ease: ${cs.flesch_reading_ease}\n`;
    }
    if (cs.flesch_kincaid_grade !== undefined) {
      formatted += `  - Flesch-Kincaid Grade: ${cs.flesch_kincaid_grade}\n`;
    }
  }

  if (scores.issue_score) {
    const is = scores.issue_score;
    if (is.grammar_score !== undefined) {
      formatted += `Grammar Score: ${is.grammar_score}`;
      if (is.grammar_issues !== undefined) {
        formatted += ` (${is.grammar_issues} issues)`;
      }
      formatted += '\n';
    }
    if (is.style_guide_score !== undefined) {
      formatted += `Style Guide Score: ${is.style_guide_score}`;
      if (is.style_guide_issues !== undefined) {
        formatted += ` (${is.style_guide_issues} issues)`;
      }
      formatted += '\n';
    }
    if (is.terminology_score !== undefined) {
      formatted += `Terminology Score: ${is.terminology_score}`;
      if (is.terminology_issues !== undefined) {
        formatted += ` (${is.terminology_issues} issues)`;
      }
      formatted += '\n';
    }
    if (is.tone_score !== undefined) {
      formatted += `Tone Score: ${is.tone_score}\n`;
      if (is.informality !== undefined && is.target_informality !== undefined) {
        formatted += `  - Informality: ${is.informality} (target: ${is.target_informality})\n`;
      }
      if (is.liveliness !== undefined && is.target_liveliness !== undefined) {
        formatted += `  - Liveliness: ${is.liveliness} (target: ${is.target_liveliness})\n`;
      }
    }
  }

  return formatted;
}

// Graceful shutdown handler
async function gracefulShutdown(server: Server) {
  log('info', 'Received shutdown signal, closing server gracefully...');
  try {
    await server.close();
    log('info', 'Server closed successfully');
    process.exit(0);
  } catch (error) {
    log('error', 'Error during shutdown', error);
    process.exit(1);
  }
}

// Main server
async function main() {
  const server = new Server(
    {
      name: 'markupai-mcp-server',
      vendor: 'markupai',
      version: '1.0.0',
      description: 'MCP server for Markup AI API text analysis and improvement'
    },
    {
      capabilities: {
        tools: {},
      },
    }
  );

  // Register shutdown handlers
  process.on('SIGTERM', () => gracefulShutdown(server));
  process.on('SIGINT', () => gracefulShutdown(server));

  // Handle list tools request
  server.setRequestHandler(ListToolsRequestSchema, async () => {
    return {
      tools: TOOLS,
    };
  });

  // Handle tool calls
  server.setRequestHandler(CallToolRequestSchema, async (request) => {
    const { name, arguments: args } = request.params;

    try {
      switch (name) {
        case 'markupai_rewrite': {
          if (!isBaseToolArgs(args)) {
            throw new Error('Invalid arguments for markupai_rewrite');
          }

          const { text, dialect = 'american_english', tone = 'professional', style_guide = 'microsoft' } = args;

          log('info', `Starting rewrite operation`, {
            textLength: text.length,
            dialect,
            tone,
            style_guide
          });

          // Submit rewrite request
          const submitResult = await submitWorkflow('rewrites', text, dialect, tone, style_guide);

          // Poll for completion if workflow is running
          const result = submitResult.status === 'running' && submitResult.workflow_id
            ? await pollWorkflowCompletion(submitResult.workflow_id, 'rewrites')
            : await getWorkflowStatus(submitResult.workflow_id, 'rewrites');

          return {
            content: [
              {
                type: 'text',
                text: formatResponse(result)
              }
            ]
          };
        }

        case 'markupai_check': {
          if (!isBaseToolArgs(args)) {
            throw new Error('Invalid arguments for markupai_check');
          }

          const { text, dialect = 'american_english', tone = 'professional', style_guide = 'microsoft' } = args;

          log('info', `Starting check operation`, {
            textLength: text.length,
            dialect,
            tone,
            style_guide
          });

          // Submit check request
          const submitResult = await submitWorkflow('checks', text, dialect, tone, style_guide);

          // Poll for completion if workflow is running
          const result = submitResult.status === 'running' && submitResult.workflow_id
            ? await pollWorkflowCompletion(submitResult.workflow_id, 'checks')
            : await getWorkflowStatus(submitResult.workflow_id, 'checks');

          return {
            content: [
              {
                type: 'text',
                text: formatResponse(result)
              }
            ]
          };
        }

        case 'markupai_suggestions': {
          if (!isBaseToolArgs(args)) {
            throw new Error('Invalid arguments for markupai_suggestions');
          }

          const { text, dialect = 'american_english', tone = 'professional', style_guide = 'microsoft' } = args;

          log('info', `Starting suggestions operation`, {
            textLength: text.length,
            dialect,
            tone,
            style_guide
          });

          // Submit suggestions request
          const submitResult = await submitWorkflow('suggestions', text, dialect, tone, style_guide);

          // Poll for completion if workflow is running
          const result = submitResult.status === 'running' && submitResult.workflow_id
            ? await pollWorkflowCompletion(submitResult.workflow_id, 'suggestions')
            : await getWorkflowStatus(submitResult.workflow_id, 'suggestions');

          return {
            content: [
              {
                type: 'text',
                text: formatResponse(result)
              }
            ]
          };
        }

        case 'markupai_workflow_status': {
          if (!isWorkflowStatusArgs(args)) {
            throw new Error('Invalid arguments for markupai_workflow_status');
          }

          const { workflow_id, workflow_type } = args;

          log('info', `Checking workflow status`, {
            workflow_id,
            workflow_type
          });

          const result = await getWorkflowStatus(workflow_id, workflow_type);

          return {
            content: [
              {
                type: 'text',
                text: formatResponse(result)
              }
            ]
          };
        }

        default:
          throw new Error(`Unknown tool: ${name}`);
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      log('error', `Tool execution failed`, {
        tool: name,
        error: errorMessage
      });

      return {
        content: [
          {
            type: 'text',
            text: `Error: ${errorMessage}`
          }
        ],
        isError: true
      };
    }
  });

  const transport = new StdioServerTransport();
  await server.connect(transport);

  log('info', 'Markup AI MCP server running on stdio');
}

main().catch((error) => {
  log('error', 'Server startup failed', error);
  console.error('Fatal error:', error);
  process.exit(1);
});
