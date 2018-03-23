/*!
* *****************************************************************************
*   \file shLighting_main.cpp
*   \author moennen
*   \brief
*   \date 2017-12-19
*   *****************************************************************************/

#include "utils/gl_utils.h"
#include "utils/gl_utils.inline.h"
#include "utils/phSpline.h"

#include "SDL2/SDL.h"
#include "SDL2/SDL_opengl.h"

// opengl
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
using namespace glm;

// opencv
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>

using namespace std;
using namespace cv;
using namespace glm;

const string keys =
    "{help h usage ? |         | print this message   }"
    "{@imageA        |         | image1 for compare   }";

static ivec2 windowSz(1024, 768);

struct UIData 
{
  // point currently selected (if any)
  size_t currSelection;
  // viewport transform
  mat4 currTransform;
  
} s_uiData;


struct SceneData 
{
  vector<vec2> s_mapPts;

} s_sceneData;

void init() {
  glClearColor(0.0, 0.0, 0.0, 0.0);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0.0, windowSz.x, windowSz.y, 1.0, -1.0, 1.0);
  glEnable(GL_BLEND);
  glEnable(GL_TEXTURE_2D);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}
void draw(GLuint tex, size_t w, size_t h) {
  // Bind Texture
  glBindTexture(GL_TEXTURE_2D, tex);

  GLfloat Vertices[] = {(float)0,
                        (float)0,
                        0,
                        (float)0 + w,
                        (float)0,
                        0,
                        (float)0 + (float)w,
                        (float)0 + (float)h,
                        0,
                        (float)0,
                        (float)0 + (float)h,
                        0};
  GLfloat TexCoord[] = {
      0, 0, 1, 0, 1, 1, 0, 1,
  };
  GLubyte indices[] = {
      0, 1,
      2,         // first triangle (bottom left - top left - top right)
      0, 2, 3};  // second triangle (bottom left - top right - bottom right)

  glEnableClientState(GL_VERTEX_ARRAY);
  glVertexPointer(3, GL_FLOAT, 0, Vertices);

  glEnableClientState(GL_TEXTURE_COORD_ARRAY);
  glTexCoordPointer(2, GL_FLOAT, 0, TexCoord);

  glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_BYTE, indices);

  glDisableClientState(GL_TEXTURE_COORD_ARRAY);
  glDisableClientState(GL_VERTEX_ARRAY);
}

int main(int argc, char* argv[]) {
  CommandLineParser parser(argc, argv, keys);
  if (parser.has("help")) {
    parser.printMessage();
    return (0);
  }
  string inputFilenameA = parser.get<string>("@imageA");

  /*auto func = [](const vec2 x) {
    return exp(0.5 * x.x * x.x) * exp(0.5 * x.y * x.y);
  };
  vector<vec2> ctrlX = {
      {0.3f, -0.1f}, {-0.04f, 0.8f}, {-0.97f, 0.001f}, {0.75f, -1.3f}};
  vector<float> ctrlY;
  ctrlY.reserve(ctrlX.size());
  transform(ctrlX.cbegin(), ctrlX.cend(), std::back_inserter(ctrlY), func);

  PhSpline<2, 1, 2, float> interpFunc(value_ptr(ctrlX[0]), &ctrlY[0],
                                      ctrlX.size());

  for (float x = -2.0f; x <= 2.0f; x += 0.01f) {
    // for (float y = -1.0f; y <= 1.0f; y += 0.1f) {
    const vec2 xy(x, 0.0f);
    const float rf = func(xy);
    float ef;
    interpFunc(value_ptr(ctrlX[0]), value_ptr(xy), &ef);
    cout << x << " " << rf << " " << ef << endl;
  }
  //}*/

  auto func = [](const float x) { return x < 0.0f ? 0.0f : exp(0.5 * x * x); };
  vector<float> ctrlX = {-2.0f, 1.0f,   1.5f, 0.3f,   -0.1f, 0.1f, 0.5f,
                         -0.6f, -0.04f, 0.8f, -1.75f, -1.3f, 2.0f};
  vector<float> ctrlY;
  ctrlY.reserve(ctrlX.size());
  transform(ctrlX.cbegin(), ctrlX.cend(), std::back_inserter(ctrlY), func);

  PhSpline<1, 1, 2, float> interpFunc(&ctrlX[0], &ctrlY[0], ctrlX.size());

  for (float x = -2.0f; x <= 2.0f; x += 0.01f) {
    const float rf = func(x);
    float ef;
    interpFunc(&ctrlX[0], &x, &ef);
    cout << x << " " << rf << " " << ef << endl;
  }

  // SDL init
  SDL_Init(SDL_INIT_EVERYTHING);
  SDL_Window* window = SDL_CreateWindow("shLighting", SDL_WINDOWPOS_CENTERED,
                                        SDL_WINDOWPOS_CENTERED, windowSz.x,
                                        windowSz.y, SDL_WINDOW_OPENGL);
  SDL_GLContext glCtx = SDL_GL_CreateContext(window);

  // load and setup the input image
  cv::Mat inputA = cv::imread(inputFilenameA.c_str());
  if (inputA.type() != CV_32F) {
    inputA.convertTo(inputA, CV_32F);
    inputA *= 1.0 / 255.0;
  }
  cv::cvtColor(inputA, inputA, cv::COLOR_BGR2RGB);

  //
  gl_utils::Texture<gl_utils::RGB_32FP> texA(
      glm::uvec2(inputA.cols, inputA.rows));
  gl_utils::uploadToTexture(texA, inputA.ptr());

  cv::Mat output(texA.sz.x, texA.sz.y, CV_32FC3);
  gl_utils::readbackTexture(texA, output.ptr());

  bool running = true;
  Uint32 start;
  SDL_Event event;

  init();
  while (running) {
    start = SDL_GetTicks();
    draw(texA.id, texA.sz.x, texA.sz.y);
    while (SDL_PollEvent(&event)) {
      switch (event.type) {
        case SDL_QUIT:
          running = false;
          break;
        case SDL_MOUSEBUTTONDOWN:
          switch (event.button.button) {
            case SDL_BUTTON_LEFT:
              break;
            case SDL_BUTTON_RIGHT:
              break;
            default:
              break;
          }
          break;
        case SDL_MOUSEMOTION: {
          int mouseX = event.motion.x;
          int mouseY = event.motion.y;
          std::stringstream ss;
          ss << "X: " << mouseX << " Y: " << mouseY;
          SDL_SetWindowTitle(window, ss.str().c_str());
        } break;
        case SDL_MOUSEWHEEL:
          if (event.wheel.y == 1)  // scroll up
          {
            // Pull up code here!
          } else if (event.wheel.y == -1)  // scroll down
          {
            // Pull down code here!
          }
          break;
      }
    }
    SDL_GL_SwapWindow(window);
    if (1000 / 60 > (SDL_GetTicks() - start))
      SDL_Delay(1000 / 60 - (SDL_GetTicks() - start));
  }
  SDL_Quit();

  return (0);
}
