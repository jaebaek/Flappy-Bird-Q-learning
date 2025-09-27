package com.simple.flappybird;

import com.badlogic.gdx.ApplicationAdapter;

/** {@link com.badlogic.gdx.ApplicationListener} implementation shared by all platforms. */
public class Main extends ApplicationAdapter {
    private FlappyBird game;

    @Override
    public void create() {
        game = new FlappyBird();
        game.create();
    }

    @Override
    public void render() {
        game.render();
    }

    @Override
    public void dispose() {
        game.dispose();
    }
}
