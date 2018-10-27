

#include <vector>

// MAPPING #0
//
// MIT SCENE PARSING DATASET 151 categories --> 6 categories
// {
//   0 : nothing
//   1 : buildings
//   2 : sky
//   3 : background (generic stuff)
//   4 : person, animals
//   5 : vehicles
// }

static const std::vector<unsigned char> om0000_mit_151_007 = {
    0,  // : 0 = Nothing
    3,  // : 1 = wall
    1,  // : 2 = building,
    2,  // : 3 = sky
    3,  // : 4 = floor,
    3,  // : 5 = tree
    3,  // : 6 = ceiling
    3,  // : 7 = road,
    3,  // : 8 = bed
    3,  // : 9 = windowpane,
    3,  // : 10 = grass
    3,  // : 11 = cabinet
    3,  // : 12 = sidewalk,
    4,  // : 13 = person,
    3,  // : 14 = earth,
    3,  // : 15 = door,
    3,  // : 16 = table
    3,  // : 17 = mountain,
    3,  // : 18 = plant,
    3,  // : 19 = curtain,
    3,  // : 20 = chair
    5,  // : 21 = car,
    3,  // : 22 = water
    3,  // : 23 = painting,
    3,  // : 24 = sofa,
    3,  // : 25 = shelf
    1,  // : 26 = house
    3,  // : 27 = sea
    3,  // : 28 = mirror
    3,  // : 29 = rug,
    3,  // : 30 = field
    3,  // : 31 = armchair
    3,  // : 32 = seat
    3,  // : 33 = fence,
    3,  // : 34 = desk
    3,  // : 35 = rock,
    3,  // : 36 = wardrobe,
    3,  // : 37 = lamp
    3,  // : 38 = bathtub,
    3,  // : 39 = railing,
    3,  // : 40 = cushion
    3,  // : 41 = base,
    3,  // : 42 = box
    3,  // : 43 = column,
    3,  // : 44 = signboard,
    3,  // : 45 = chest
    3,  // : 46 = counter
    3,  // : 47 = sand
    3,  // : 48 = sink
    1,  // : 49 = skyscraper
    3,  // : 50 = fireplace,
    3,  // : 51 = refrigerator,
    3,  // : 52 = grandstand,
    3,  // : 53 = path
    3,  // : 54 = stairs,
    3,  // : 55 = runway
    3,  // : 56 = case,
    3,  // : 57 = pool
    3,  // : 58 = pillow
    3,  // : 59 = screen
    3,  // : 60 = stairway,
    3,  // : 61 = river
    3,  // : 62 = bridge,
    3,  // : 63 = bookcase
    3,  // : 64 = blind,
    3,  // : 65 = coffee
    3,  // : 66 = toilet,
    3,  // : 67 = flower
    3,  // : 68 = book
    3,  // : 69 = hill
    3,  // : 70 = bench
    3,  // : 71 = countertop
    3,  // : 72 = stove,
    3,  // : 73 = palm,
    3,  // : 74 = kitchen
    3,  // : 75 = computer,
    3,  // : 76 = swivel
    3,  // : 77 = boat
    3,  // : 78 = bar
    3,  // : 79 = arcade
    3,  // : 80 = hovel,
    5,  // : 81 = bus,
    3,  // : 82 = towel
    3,  // : 83 = light,
    5,  // : 84 = truck,
    1,  // : 85 = tower
    3,  // : 86 = chandelier,
    3,  // : 87 = awning,
    3,  // : 88 = streetlight,
    3,  // : 89 = booth,
    3,  // : 90 = television
    5,  // : 91 = airplane,
    3,  // : 92 = dirt
    3,  // : 93 = apparel,
    3,  // : 94 = pole
    3,  // : 95 = land,
    3,  // : 96 = bannister,
    3,  // : 97 = escalator,
    3,  // : 98 = ottoman,
    3,  // : 99 = bottle
    3,  // : 100 = buffet,
    3,  // : 101 = poster,
    3,  // : 102 = stage
    5,  // : 103 = van
    5,  // : 104 = ship
    3,  // : 105 = fountain
    3,  // : 106 = conveyer
    3,  // : 107 = canopy
    3,  // : 108 = washer,
    3,  // : 109 = plaything,
    3,  // : 110 = swimming
    3,  // : 111 = stool
    3,  // : 112 = barrel,
    3,  // : 113 = basket,
    3,  // : 114 = waterfall,
    3,  // : 115 = tent,
    3,  // : 116 = bag
    3,  // : 117 = minibike,
    3,  // : 118 = cradle
    3,  // : 119 = oven
    3,  // : 120 = ball
    3,  // : 121 = food,
    3,  // : 122 = step,
    3,  // : 123 = tank,
    3,  // : 124 = trade
    3,  // : 125 = microwave,
    3,  // : 126 = pot,
    4,  // : 127 = animal,
    5,  // : 128 = bicycle,
    3,  // : 129 = lake
    3,  // : 130 = dishwasher,
    3,  // : 131 = screen,
    3,  // : 132 = blanket,
    3,  // : 133 = sculpture
    3,  // : 134 = hood,
    3,  // : 135 = sconce
    3,  // : 136 = vase
    5,  // : 137 = traffic
    3,  // : 138 = tray
    3,  // : 139 = ashcan,
    3,  // : 140 = fan
    3,  // : 141 = pier,
    3,  // : 142 = crt
    3,  // : 143 = plate
    3,  // : 144 = monitor,
    3,  // : 145 = bulletin
    3,  // : 146 = shower
    3,  // : 147 = radiator
    3,  // : 148 = glass,
    3,  // : 149 = clock
    3   // : 150 = flag
};

// MAPPING #1
//
// MIT SCENE PARSING DATASET 151 categories --> 2 categories
// {
//   0 : all
//   1 : sky
// }

static const std::vector<unsigned char> om0001_mit_151_002 = {
    0,  // : 0 = Nothing
    0,  // : 1 = wall
    0,  // : 2 = building,
    1,  // : 3 = sky
    0,  // : 4 = floor,
    0,  // : 5 = tree
    0,  // : 6 = ceiling
    0,  // : 7 = road,
    0,  // : 8 = bed
    0,  // : 9 = windowpane,
    0,  // : 10 = grass
    0,  // : 11 = cabinet
    0,  // : 12 = sidewalk,
    0,  // : 13 = person,
    0,  // : 14 = earth,
    0,  // : 15 = door,
    0,  // : 16 = table
    0,  // : 17 = mountain,
    0,  // : 18 = plant,
    0,  // : 19 = curtain,
    0,  // : 20 = chair
    0,  // : 21 = car,
    0,  // : 22 = water
    0,  // : 23 = painting,
    0,  // : 24 = sofa,
    0,  // : 25 = shelf
    0,  // : 26 = house
    0,  // : 27 = sea
    0,  // : 28 = mirror
    0,  // : 29 = rug,
    0,  // : 30 = field
    0,  // : 31 = armchair
    0,  // : 32 = seat
    0,  // : 33 = fence,
    0,  // : 34 = desk
    0,  // : 35 = rock,
    0,  // : 36 = wardrobe,
    0,  // : 37 = lamp
    0,  // : 38 = bathtub,
    0,  // : 39 = railing,
    0,  // : 40 = cushion
    0,  // : 41 = base,
    0,  // : 42 = box
    0,  // : 43 = column,
    0,  // : 44 = signboard,
    0,  // : 45 = chest
    0,  // : 46 = counter
    0,  // : 47 = sand
    0,  // : 48 = sink
    0,  // : 49 = skyscraper
    0,  // : 50 = fireplace,
    0,  // : 51 = refrigerator,
    0,  // : 52 = grandstand,
    0,  // : 53 = path
    0,  // : 54 = stairs,
    0,  // : 55 = runway
    0,  // : 56 = case,
    0,  // : 57 = pool
    0,  // : 58 = pillow
    0,  // : 59 = screen
    0,  // : 60 = stairway,
    0,  // : 61 = river
    0,  // : 62 = bridge,
    0,  // : 63 = bookcase
    0,  // : 64 = blind,
    0,  // : 65 = coffee
    0,  // : 66 = toilet,
    0,  // : 67 = flower
    0,  // : 68 = book
    0,  // : 69 = hill
    0,  // : 70 = bench
    0,  // : 71 = countertop
    0,  // : 72 = stove,
    0,  // : 73 = palm,
    0,  // : 74 = kitchen
    0,  // : 75 = computer,
    0,  // : 76 = swivel
    0,  // : 77 = boat
    0,  // : 78 = bar
    0,  // : 79 = arcade
    0,  // : 80 = hovel,
    0,  // : 81 = bus,
    0,  // : 82 = towel
    0,  // : 83 = light,
    0,  // : 84 = truck,
    0,  // : 85 = tower
    0,  // : 86 = chandelier,
    0,  // : 87 = awning,
    0,  // : 88 = streetlight,
    0,  // : 89 = booth,
    0,  // : 90 = television
    0,  // : 91 = airplane,
    0,  // : 92 = dirt
    0,  // : 93 = apparel,
    0,  // : 94 = pole
    0,  // : 95 = land,
    0,  // : 96 = bannister,
    0,  // : 97 = escalator,
    0,  // : 98 = ottoman,
    0,  // : 99 = bottle
    0,  // : 100 = buffet,
    0,  // : 101 = poster,
    0,  // : 102 = stage
    0,  // : 103 = van
    0,  // : 104 = ship
    0,  // : 105 = fountain
    0,  // : 106 = conveyer
    0,  // : 107 = canopy
    0,  // : 108 = washer,
    0,  // : 109 = plaything,
    0,  // : 110 = swimming
    0,  // : 111 = stool
    0,  // : 112 = barrel,
    0,  // : 113 = basket,
    0,  // : 114 = waterfall,
    0,  // : 115 = tent,
    0,  // : 116 = bag
    0,  // : 117 = minibike,
    0,  // : 118 = cradle
    0,  // : 119 = oven
    0,  // : 120 = ball
    0,  // : 121 = food,
    0,  // : 122 = step,
    0,  // : 123 = tank,
    0,  // : 124 = trade
    0,  // : 125 = microwave,
    0,  // : 126 = pot,
    0,  // : 127 = animal,
    0,  // : 128 = bicycle,
    0,  // : 129 = lake
    0,  // : 130 = dishwasher,
    0,  // : 131 = screen,
    0,  // : 132 = blanket,
    0,  // : 133 = sculpture
    0,  // : 134 = hood,
    0,  // : 135 = sconce
    0,  // : 136 = vase
    0,  // : 137 = traffic
    0,  // : 138 = tray
    0,  // : 139 = ashcan,
    0,  // : 140 = fan
    0,  // : 141 = pier,
    0,  // : 142 = crt
    0,  // : 143 = plate
    0,  // : 144 = monitor,
    0,  // : 145 = bulletin
    0,  // : 146 = shower
    0,  // : 147 = radiator
    0,  // : 148 = glass,
    0,  // : 149 = clock
    0   // : 150 = flag
};

// MAPPING #2
//
// COCO SCENE PARSING DATASET 181 categories --> 8 categories
// {
//   0 : unlabeled
//   1 : background artificial
//   2 : sky
//   3 : background natural
//   4 : person
//   5 : vehicles
//   6 : animals
//   7 : stuff   
// }
static const std::vector<unsigned char> om0000_coco_183_007 = {
    0,  // unlabeled
    4,  // person
    5,  // bicycle
    5,  // car
    5,  // motorcycle
    5,  // airplane
    5,  // bus
    5,  // train
    5,  // truck
    5,  // boat
    7,  // traffic light
    7,  // fire hydrant
    7,  // street sign
    7,  // stop sign
    7,  // parking meter
    7,  // bench
    6,  // bird
    6,  // cat
    6,  // dog
    6,  // horse
    6,  // sheep
    6,  // cow
    6,  // elephant
    6,  // bear
    6,  // zebra
    6,  // giraffe
    0,  // hat
    7,  // backpack
    7,  // umbrella
    0,  // shoe
    0,  // eye glasses
    7,  // handbag
    0,  // tie
    7,  // suitcase
    7,  // frisbee
    7,  // skis
    7,  // snowboard
    7,  // sports ball
    7,  // kite
    7,  // baseball bat
    7,  // baseball glove
    7,  // skateboard
    7,  // surfboard
    7,  // tennis racket
    7,  // bottle
    7,  // plate
    7,  // wine glass
    7,  // cup
    7,  // fork
    7,  // knife
    7,  // spoon
    7,  // bowl
    7,  // banana
    7,  // apple
    7,  // sandwich
    7,  // orange
    7,  // broccoli
    7,  // carrot
    7,  // hot dog
    7,  // pizza
    7,  // donut
    7,  // cake
    7,  // chair
    7,  // couch
    7,  // potted plant
    7,  // bed
    7,  // mirror
    7,  // dining table
    1,  // window
    7,  // desk
    7,  // toilet
    1,  // door
    7,  // tv
    7,  // laptop
    7,  // mouse
    7,  // remote
    7,  // keyboard
    7,  // cell phone
    7,  // microwave
    7,  // oven
    7,  // toaster
    7,  // sink
    7,  // refrigerator
    7,  // blender
    7,  // book
    7,  // clock
    7,  // vase
    7,  // scissors
    7,  // teddy bear
    7,  // hair drier
    7,  // toothbrush
    7,  // hair brush
    7,  // banner
    7,  // blanket
    3,  // branch
    1,  // bridge
    1,  // building-other
    3,  // bush
    7,  // cabinet
    7,  // cage
    7,  // cardboard
    7,  // carpet
    1,  // ceiling-other
    1,  // ceiling-tile
    0,  // cloth
    0,  // clothes
    2,  // clouds
    7,  // counter
    7,  // cupboard
    1,  // curtain
    7,  // desk-stuff
    7,  // dirt
    0,  // door-stuff
    0,  // fence
    1,  // floor-marble
    1,  // floor-other
    1,  // floor-stone
    1,  // floor-tile
    1,  // floor-wood
    3,  // flower
    3,  // fog
    7,  // food-other
    7,  // fruit
    7,  // furniture-other
    3,  // grass
    3,  // gravel
    3,  // ground-other
    3,  // hill
    1,  // house
    3,  // leaves
    0,  // light
    7,  // mat
    0,  // metal
    0,  // mirror-stuff
    3,  // moss
    3,  // mountain
    3,  // mud
    7,  // napkin
    7,  // net
    7,  // paper
    1,  // pavement
    7,  // pillow
    3,  // plant-other
    7,  // plastic
    1,  // platform
    0,  // playingfield
    0,  // railing
    1,  // railroad
    3,  // river
    1,  // road
    3,  // rock
    1,  // roof
    0,  // rug
    7,  // salad
    3,  // sand
    3,  // sea
    7,  // shelf
    2,  // sky-other
    1,  // skyscraper
    3,  // snow
    0,  // solid-other
    1,  // stairs
    3,  // stone
    3,  // straw
    1,  // structural-other
    7,  // table
    7,  // tent
    0,  // textile-other
    0,  // towel
    3,  // tree
    7,  // vegetable
    1,  // wall-brick
    1,  // wall-concrete
    1,  // wall-other
    1,  // wall-panel
    1,  // wall-stone
    1,  // wall-tile
    1,  // wall-wood
    3,  // water-other
    0,  // waterdrops
    0,  // window-blind
    1,  // window-other
    3,  // wood
};

// return the mapping given its ID
static const std::vector<unsigned char> getObjectsMapping( int id )
{
   return id == 0 ? om0000_mit_151_007 : ( id == 1 ? om0001_mit_151_002 : om0000_coco_183_007 );
}
